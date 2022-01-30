#include "Timer.h"

#include <iostream>

#include "Utils/Log.h"

Timer::Timer(const char* label /* = nullptr */)
   : m_Label(label), m_Start(clock_t::now()) {}

Timer::~Timer()
{
   stop();
}

void Timer::stop()
{
   if (m_Stopped)
      return;
   m_Stopped = true;

   float ms = elapsed();
   if (m_Label != nullptr)
      print("[", m_Label, "] ", ms, " ms");
   else
      print(ms, " ms");
}

float Timer::elapsed()
{
   clock_t::time_point end = clock_t::now();
   return std::chrono::duration<float>(end - m_Start).count() * 1000;
}
