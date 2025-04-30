#!/bin/bash
# as of the 2025/04/29, we need to use this list of nodes if we want to be able to do a proper profiling
salloc --nodelist=mel[2012,2016-2020,2022,2025-2027,2031,2036,2047,2050-2052,2054,2056,2058,2059,2062,2063,2068,2073,2082-2084,2090,2092,2095,2106,2108,2114-2116,2123-2126,2128,2132,2140,2147,2154,2155,2157,2158,2160,2161,2179,2181-2183,2185,2187,2191,2192,2198-2200] -p gpu --qos default -A p200865 -N 1 -t 8:00:0 --disable-perfparanoid 

#--reservation=scynergy-profiling-ws


