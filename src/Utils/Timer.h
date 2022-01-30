#pragma once

#include <chrono>

class Timer
{
public:
   using clock_t = std::chrono::high_resolution_clock;

   Timer(const char* label = nullptr);
   ~Timer();

   void stop();
   float elapsed();

private:
   bool m_Stopped = false;
   const char* m_Label;
   clock_t::time_point m_Start;
};
