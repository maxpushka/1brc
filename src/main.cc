#include <atomic>
#include <charconv>
#include <iostream>
#include <limits>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string>
#include <string_view>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <execution>
#include <print>

constexpr size_t UNIQUE_NAMES = 10000;

namespace {
class MappedFile {
  int fd = -1;
  void *addr = nullptr;
  size_t fileSize = 0;
 public:
  MappedFile() = default;

  explicit MappedFile(const std::string &filename) {
    fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
      throw std::runtime_error("Failed to open file");
    }

    // Obtain the size of the file
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
      close(fd);
      throw std::runtime_error("Failed to get file size");
    }
    fileSize = sb.st_size;

    // Map the file into memory
    addr = mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
      close(fd);
      throw std::runtime_error("Failed to map the file");
    }
  }

  ~MappedFile() {
    if (addr) munmap(addr, fileSize);
    if (fd != -1) close(fd);
  }

  MappedFile(const MappedFile &) = delete;
  MappedFile &operator=(const MappedFile &) = delete;

  MappedFile(MappedFile &&other) noexcept: fd(other.fd), addr(other.addr), fileSize(other.fileSize) {
    other.addr = nullptr;
    other.fd = -1;
    other.fileSize = 0;
  }

  MappedFile &operator=(MappedFile &&other) noexcept {
    if (this == &other) return *this;
    addr = other.addr;
    fileSize = other.fileSize;
    fd = other.fd;

    other.addr = nullptr;
    other.fileSize = 0;
    other.fd = -1;
    return *this;
  }

  [[nodiscard]] const char *data() const &{
    return reinterpret_cast<const char *>(addr);
  }

  [[nodiscard]] size_t size() const &{ return fileSize; }
};

struct StationData {
  std::atomic<float> min = std::numeric_limits<float>::max();
  std::atomic<float> max = std::numeric_limits<float>::min();
  std::atomic<uint> count = 0;
  std::atomic<float> sum = 0;

  std::atomic<int> index = -1;
  std::string_view name;
  std::mutex name_mutex;

  void set_name(const std::string_view& station, const size_t station_index) {
    // Early check to reduce locking overhead
    if (index.load(std::memory_order_acquire) != -1) return;

    std::lock_guard<std::mutex> lock(name_mutex);
    // Confirm the condition hasn't changed after acquiring the lock
    if (index.load(std::memory_order_relaxed) != -1) return;

    name = station;
    index.store(station_index, std::memory_order_release);
  }
};

std::array<StationData, UNIQUE_NAMES> stations{};

void process_line(const std::string_view &line) {
  size_t pos = line.find(';');
  // No need to check for missing ';'
  // since the data is assumed to be well-formed

  std::string_view station = line.substr(0, pos);
  size_t index = std::hash<std::string_view>{}(station) % UNIQUE_NAMES;

  std::string_view measurement = line.substr(pos + 1);
  float temperature = 0.0f;
  auto
      [_, ec] = std::from_chars(measurement.data(), measurement.data() + measurement.size(), temperature);
  if (ec != std::errc()) {
    std::cerr << "Error: failed to parse temperature (" << measurement << ')' << std::endl;
    throw 1;
  }

  StationData &it = stations.at(index);

  float prevMin = it.min.load();
  while (temperature < prevMin) {
    if (it.min.compare_exchange_weak(prevMin, temperature)) break;
    prevMin = it.min.load();
  }

  float prevMax = it.max.load();
  while (temperature > prevMax) {
    if (it.max.compare_exchange_weak(prevMax, temperature)) break;
    prevMax = it.max.load();
  }

  it.count.fetch_add(1);
  it.sum.fetch_add(temperature);

  it.set_name(std::move(station), index);
}

#if defined(USE_NAIVE)
std::vector<std::string_view> split_by_rows(const char *data, const size_t size) {
  std::vector<std::string_view> result;

  size_t start = 0;
  for (size_t i = 0; i < size; ++i) {
    if (data[i] != '\n') continue;
    result.emplace_back(data + start, i - start);
    start = i + 1;
  }

  // Handle the last substring
  if (start < size) {
    result.emplace_back(data + start, size - start);
  }

  return result;
}
#else
#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
// Macros definitions for SIMD operations based on compiler flags
#if defined(USE_AVX512)
#define SIMD_TYPE __m512i
#define SET_NEWLINE _mm512_set1_epi8
#define LOAD_SI _mm512_loadu_si512
#define MOVE_MASK(block, newline) _mm512_cmpeq_epi8_mask(block, newline)
#define SIMD_TZCNT _tzcnt_u64
#define SIMD_WIDTH 64
#elif defined(USE_AVX2)
#define SIMD_TYPE __m256i
#define SET_NEWLINE _mm256_set1_epi8
#define LOAD_SI _mm256_loadu_si256
#define MOVE_MASK(block, newline) _mm256_movemask_epi8(_mm256_cmpeq_epi8(block, newline))
#define SIMD_TZCNT _tzcnt_u32
#define SIMD_WIDTH 32
#elif defined(USE_SSE2)
#define SIMD_TYPE __m128i
#define SET_NEWLINE _mm_set1_epi8
#define LOAD_SI _mm_loadu_si128
#define MOVE_MASK(block, newline) _mm_movemask_epi8(_mm_cmpeq_epi8(block, newline))
#define SIMD_TZCNT _tzcnt_u32
#define SIMD_WIDTH 16
#endif

std::vector<std::string_view> split_by_rows(const char *data, size_t size) {
  std::vector<std::string_view> result;
  const char *start = data;
  SIMD_TYPE newline = SET_NEWLINE('\n');

  // Process data in chunks of SIMD_WIDTH
  while (size >= SIMD_WIDTH) {
    SIMD_TYPE block = LOAD_SI(reinterpret_cast<const SIMD_TYPE *>(data));
    auto mask = MOVE_MASK(block, newline);

    while (mask != 0) {
      int bit = SIMD_TZCNT(mask);
      result.emplace_back(start, data + bit);
      start = data + bit + 1;
      mask &= mask - 1;  // Clear the lowest set bit
    }

    data += SIMD_WIDTH;
    size -= SIMD_WIDTH;
  }

  // Handle any remaining characters
  while (size > 0) {
    if (*data == '\n') {
      result.emplace_back(start, data);
      start = data + 1;
    }
    data++;
    size--;
  }

  // Add the last piece if there's no newline at the end
  if (start != data) {
    result.emplace_back(start, data);
  }

  return result;
}
#endif
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Error: provide absolute path to dataset" << std::endl;
    return 1;
  }

  MappedFile file;
  try {
    file = MappedFile{argv[1]};
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  auto rows = split_by_rows(file.data(), file.size());
  std::println("Finished parsing file");
  std::for_each(std::execution::par_unseq, rows.begin(), rows.end(), [](std::string_view row) {
    process_line(row);
  });

  std::cout << std::fixed << std::setprecision(1) << "{";
  for (size_t i = 0; const auto &station : stations) {
    if (station.count == 0) continue;

    if (i != 0) std::cout << ", ";
    std::cout << station.name << '='  // '(' << station.index << ")="
      << station.min << '/'
      << station.max << '/'
      << station.sum / static_cast<float>(station.count);
    ++i;
  }
  std::cout << "}" << std::endl;

  return 0;
}
