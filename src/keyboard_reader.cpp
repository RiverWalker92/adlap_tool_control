#include "adlap_tool_control/keyboard_reader.hpp"
#include <unistd.h>
#include <stdexcept>
#include <cstring>

KeyboardReader::KeyboardReader() : file_descriptor_(0)
{
  // get the console in raw mode
  tcgetattr(file_descriptor_, &cooked_);
  struct termios raw;
  memcpy(&raw, &cooked_, sizeof(struct termios));
  raw.c_lflag &= ~(ICANON | ECHO);
  // Setting a new line, then end of file
  raw.c_cc[VEOL] = 1;
  raw.c_cc[VEOF] = 2;
  tcsetattr(file_descriptor_, TCSANOW, &raw);
}

void KeyboardReader::readOne(char* c)
{
  int rc = read(file_descriptor_, c, 1);
  if (rc < 0)
  {
    throw std::runtime_error("read failed");
  }
}

void KeyboardReader::shutdown()
{
  tcsetattr(file_descriptor_, TCSANOW, &cooked_);
}
