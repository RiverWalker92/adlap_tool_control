#pragma once

#include <string>

class SerialPort {
public:
    SerialPort(const std::string& device, int baudrate);
    ~SerialPort();

    bool open_port();
    void close_port();
    bool write_data(const std::string& data);
    std::string read_data(size_t max_bytes = 100);
    bool is_open() const { return fd_ != -1; }

private:
    std::string device_;
    int baudrate_;
    int fd_;
};
