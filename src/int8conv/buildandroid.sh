

ABI="armeabi-v7a with NEON"


#新建build目录
mkdir build
cd build
cmake ~/git/cpuGEMM/src/int8conv \
      -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DANDROID_ABI="${ABI}" \
      -DANDROID_STL=c++_static \
      -DANDROID_NATIVE_API_LEVEL=android-21  \
      -DANDROID_TOOLCHAIN=clang \
      -DBUILD_FOR_ANDROID_COMMAND=true \
      -DNATIVE_LIBRARY_OUTPUT=. \
      -DDEBUG=OFF
#      -DRAPIDNET_BENCHMARK_LAYER=ON
make -j4
cp test ~/git/tmp/rpn
