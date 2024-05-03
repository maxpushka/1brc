#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <string>
#include <queue>
#include <thread>
#include <condition_variable>

#include "thread_pool.h"

namespace {
class MappedFile {
  int fd = -1;
  void *addr = nullptr;
  size_t fileSize = 0;
 public:
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
    if (this != &other) return *this;
    addr = other.addr;
    fileSize = other.fileSize;
    fd = other.fd;

    other.addr = nullptr;
    other.fileSize = 0;
    other.fd = -1;
    return *this;
  }

  [[nodiscard]] void *data() const &{ return addr; }

  [[nodiscard]] size_t size() const &{ return fileSize; }
};

struct StationData {
  float min = std::numeric_limits<float>::max();
  float max = std::numeric_limits<float>::min();
  uint count = 0;
  float mean = 0;
};

thread_pool::ThreadPool pool(std::thread::hardware_concurrency());
std::mutex mtx;
std::unordered_map<std::string, StationData> station_map;

void process_line(const std::string &line) {
  if (line.empty() || line[0] == '#') return;  // Skip empty lines and comments

  size_t pos = line.find(';'); // Each entry follows
  if (pos == std::string::npos) return; // Missing ';' in the line, invalid format

  std::string station = line.substr(0, pos);
  float temperature = std::stof(line.substr(pos + 1));

  std::lock_guard<std::mutex> lock(mtx);

  auto it = station_map.find(station);
  if (it != station_map.end()) {
    it->second.min = std::min(it->second.min, temperature);
    it->second.max = std::max(it->second.max, temperature);
    it->second.count++;
    it->second.mean = (temperature - it->second.mean) / static_cast<float>(it->second.count);
  } else {
    // Insert a new station
    StationData new_station_data = {
        .min = temperature,
        .max = temperature,
        .count = 1,
        .mean = temperature
    };
    station_map.emplace(station, new_station_data);
  }
}

void dispatch_rows(const char *data, size_t size) {
  size_t start = 0;
  for (size_t i = 0; i < size; ++i) {
    if (data[i] == '\n') {
      std::string line(data + start, i - start);
      pool.enqueue([line] { process_line(line); });
      start = i + 1;
    }
  }

  // Handle the last substring
  if (start < size) {
    std::string line(data + start, size - start);
    pool.enqueue([line] { process_line(line); });
  }
}
}

/*
 * TODO: potential improvements:
 *   - Use SIMD to search for '\n' in the data buffer
 *   - Pass `dispatch_rows` the start and end index of the data entry buffer instead of the entire buffer
 *   - Reduce contention
 *     - Use lock per station
 *     - Use a lock-free data structure to store the station data
 *   - Implement perfect hash function for station names to speed up lookups. Each station name is no longer than 100 bytes
 *   - Execute reduce operation on the station data in parallel. Use streaming.
 */
int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Error: provide absolute path to dataset" << std::endl;
    return 1;
  }

  try {
    MappedFile file(argv[1]);
    dispatch_rows(reinterpret_cast<char *>(file.data()), file.size());
    pool.wait_until_empty();

    for (const auto &[station, data] : station_map) {
      std::cout << station << ": " << "\n"
                << "\tMin: " << data.min << "\n"
                << "\tMax: " << data.max << "\n"
                << "\tAvg: " << data.mean << "\n";
    }
    std::cout << "Data size" << ": " << station_map.size() << std::endl;
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}
