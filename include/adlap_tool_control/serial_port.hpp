#pragma once

#include <string>

class SerialPort {
public:
    SerialPort(const std::string& device, int baudrate);
    ~SerialPort();

    bool openPort();
    void closePort();
    bool writeData(const std::string& data);
    std::string readData(size_t max_bytes = 100);
    bool isOpen() const { return fd_ != -1; }

private:
    std::string device_;
    int baudrate_;
    int fd_;
};
