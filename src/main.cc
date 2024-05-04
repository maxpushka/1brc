#include <charconv>
#include <iostream>
#include <memory>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <iomanip>
#include <vector>

#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

#include "thread_pool.h"

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
  float min = std::numeric_limits<float>::max();
  float max = std::numeric_limits<float>::min();
  uint count = 0;
  float sum = 0;
  std::shared_ptr<std::mutex> mtx = std::make_shared<std::mutex>();

  StationData(float temperature) : min(temperature), max(temperature), count(1), sum(temperature) {}
};

thread_pool::ThreadPool pool(std::thread::hardware_concurrency());
std::unordered_map<std::string_view, StationData> station_map;
std::mutex map_access_mutex;

void process_line(const std::string_view &line) {
  size_t pos = line.find(';');
  // No need to check for missing ';'
  // since the data is assumed to be well-formed

  std::string_view station = line.substr(0, pos);
  float temperature;
  std::from_chars(line.data() + pos, line.data() + pos + 1, temperature);
  // temperature = std::stof(line.substr(pos + 1));

  if (!station_map.contains(station)) {
    std::lock_guard<std::mutex> lock(map_access_mutex);
    if (!station_map.contains(station)) {
      station_map.insert({station, std::move(StationData(temperature))});
      return;
    }
  }

  std::lock_guard<std::mutex> lock(*station_map.at(station).mtx);

  StationData &it = station_map.at(station);
  it.min = std::min(it.min, temperature);
  it.max = std::max(it.max, temperature);
  it.count++;
  it.sum += temperature;
}

std::vector<std::string_view> split_by_rows_naive(const char *data, const size_t size) {
  std::vector<std::string_view> result;

  size_t start = 0;
  for (size_t i = 0; i < size; ++i) {
    if (data[i] == '\n') {
      result.emplace_back(data + start, i - start);
      start = i + 1;
    }
  }

  // Handle the last substring
  if (start < size) {
    result.emplace_back(data + start, size - start);
  }

  return result;
}

// Define macros for SIMD operations based on compiler flags
#if defined(__AVX512F__)
    #define SIMD_TYPE __m512i
    #define SET_NEWLINE _mm512_set1_epi8
    #define LOAD_SI _mm512_loadu_si512
    #define MOVE_MASK(block, newline) _mm512_cmpeq_epi8_mask(block, newline)
    #define SIMD_TZCNT _tzcnt_u64
    #define SIMD_WIDTH 64
#elif defined(__AVX2__)
    #define SIMD_TYPE __m256i
    #define SET_NEWLINE _mm256_set1_epi8
    #define LOAD_SI _mm256_loadu_si256
    #define MOVE_MASK(block, newline) _mm256_movemask_epi8(_mm256_cmpeq_epi8(block, newline))
    #define SIMD_TZCNT _tzcnt_u32
    #define SIMD_WIDTH 32
#else  // Default to SSE2
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
}

/*
 * TODO: potential improvements:
 *   - Pass `dispatch_rows` the start and end index of the data entry buffer instead of the entire buffer
 *   - Reduce contention
 *     - Use a lock-free data structure to store the station data
 *   - Implement perfect hash function for station names to speed up lookups. Each station name is no longer than 100 bytes
 *   - Execute reduce operation on the station data in parallel. Use streaming.
 */
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

  // Reserve space for the map to avoid rehashing.
  // Rehashing in concurrent environment can lead to SEGFAULT.
  station_map.reserve(10000);

  auto results = split_by_rows(file.data(), file.size());
  std::cout << "AVX512 results: " << results.size() << std::endl << results.at(0) << std::endl;

  pool.wait_until_empty();

  size_t i = 0;
  std::cout << std::fixed << std::setprecision(2) << "{";
  for (const auto &[station, data] : station_map) {
    std::cout << station << "=" << data.min << "/" << data.max << "/" << data.sum / static_cast<float>(data.count);
    if (i < station_map.size() - 1) std::cout << ", ";
    ++i;
  }
  std::cout << "}" << std::endl;

  return 0;
}
