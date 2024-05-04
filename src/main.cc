#include <charconv>
#include <iostream>
#include <memory>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <iomanip>

#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <vector>

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

std::vector<std::string_view> split_by_rows_sse2(const char *data, size_t size) {
  std::vector<std::string_view> result;
  const char *start = data;
  __m128i newline = _mm_set1_epi8('\n');

  // Process 16 bytes at a time
  while (size >= 16) {
    __m128i block = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data));
    int mask = _mm_movemask_epi8(_mm_cmpeq_epi8(block, newline));

    while (mask != 0) {
      int bit = _tzcnt_u32(mask); // Find the first set bit
      result.emplace_back(start, data + bit); // Create a string from start to the newline
      start = data + bit + 1; // Move start to after the newline
      mask &= mask - 1; // Clear the lowest set bit
    }

    data += 16;
    size -= 16;
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

std::vector<std::string_view> split_by_rows_avx2(const char *data, size_t size) {
  std::vector<std::string_view> result;
  const char *start = data;
  __m256i newline = _mm256_set1_epi8('\n');

  // Process 32 bytes at a time
  while (size >= 32) {
    __m256i block = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data));
    int mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(block, newline));

    while (mask != 0) {
      int bit = _tzcnt_u32(mask); // Find the first set bit
      result.emplace_back(start, data + bit); // Create a string from start to the newline
      start = data + bit + 1; // Move start to after the newline
      mask &= mask - 1; // Clear the lowest set bit
    }

    data += 32;
    size -= 32;
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

std::vector<std::string_view> split_by_rows_avx512(const char *data, size_t size) {
  std::vector<std::string_view> result;
  const char *start = data;
  __m512i newline = _mm512_set1_epi8('\n');

  // Process 64 bytes at a time
  while (size >= 64) {
    __m512i block = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(data));
    uint64_t mask = _mm512_cmpeq_epi8_mask(block, newline);

    while (mask != 0) {
      int bit = _tzcnt_u64(mask); // Find the first set bit
      result.emplace_back(start, data + bit); // Create a string from start to the newline
      start = data + bit + 1; // Move start to after the newline
      mask &= mask - 1; // Clear the lowest set bit
    }

    data += 64;
    size -= 64;
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

  // auto results = split_by_rows_naive(file.data(), file.size());
  // auto results = split_by_rows_sse2(file.data(), file.size());
  // auto results = split_by_rows_avx2(file.data(), file.size());
  auto results = split_by_rows_avx512(file.data(), file.size());
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
