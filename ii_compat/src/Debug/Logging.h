#pragma once

#include <iostream>

#define RESET   "\033[0m"
#define BLACK   "\033[30m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define WHITE   "\033[37m"

#define PRINT_TAG ""
#define LOG_TAG WHITE "[LOG] " RESET
#define WARNING_TAG YELLOW "[WARNING] " RESET
#define ERROR_TAG RED "[ERROR] " RESET

#define LOG(...) log(__VA_ARGS__)
#define WARNING(...) warning(__VA_ARGS__)
#define ERROR(...) error(__VA_ARGS__)

enum class OutputCode
{
	NONE,
	PRINT,
	LOG,
	WARNING,
	ERROR
};

extern OutputCode lastCode;

struct OutputParams
{
	OutputCode code;
	const char* tag;
	std::ostream& out;
};

inline void _output(std::ostream& out)
{
	out << std::endl;
}

template<class First, class... Args>
inline void _output(std::ostream& out, First&& first, Args&&... args)
{
	out << std::forward<First>(first);
	_output(out, std::forward<Args>(args)...);
}


template<class... Args>
inline constexpr void output(const OutputParams& params, Args&&... args)
{
	if (lastCode != params.code)
	{
		if (lastCode != OutputCode::NONE)
			params.out << '\n';
		lastCode = params.code;
	}
	params.out << params.tag;
	_output(params.out, std::forward<Args>(args)...);
}

template<class... Args>
inline constexpr void log(Args&&... args)
{
	OutputParams params{ OutputCode::LOG, LOG_TAG, std::cerr };
	output(params, std::forward<Args>(args)...);
}

template<class... Args>
inline constexpr void warning(Args&&... args)
{
	OutputParams params{ OutputCode::WARNING, WARNING_TAG, std::cerr };
	output(params, std::forward<Args>(args)...);
}

template<class... Args>
inline constexpr void error(Args&&... args)
{
	OutputParams params{ OutputCode::ERROR, ERROR_TAG, std::cerr };
	output(params, std::forward<Args>(args)...);
	exit(EXIT_FAILURE);
}

template<class... Args>
inline constexpr void print(Args&&... args)
{
	OutputParams params{ OutputCode::PRINT, PRINT_TAG, std::cout };
	output(params, std::forward<Args>(args)...);
}
