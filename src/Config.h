#pragma once

#include <cstdint>

enum class DatasetType
{
	DISC, CIRCLE, RING
};

struct Config
{
	DatasetType dataset_type;
	uint32_t seed;
	bool is_host_mem;
};
