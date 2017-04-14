@setlocal

cd _build_vs
cmake --build .

cd Debug\
particle_filter.exe

@endlocal