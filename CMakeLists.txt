# create a confirmation using
#    cmake -DCMAKE_BUILD_TYPE=Release .
# This is the root ITK CMakeLists file.
cmake_minimum_required(VERSION 2.8.9)
if(COMMAND CMAKE_POLICY)
  cmake_policy(SET CMP0003 NEW)
endif()

#set(ITK_DIR /Users/hauke/src/itk/InsightToolkit-4.8.1/bin)
if (EXISTS "/Users/hauke/src/itk/InsightToolkit-5.0.0/bin")
  set(ITK_DIR /Users/hauke/src/itk/InsightToolkit-5.0.0/bin)
else()
  set(ITK_DIR /opt/InsightToolkit-5.0.1/bin)
endif()

#SET(CMAKE_CXX_FLAGS "-std=c++0x")
#SET(CMAKE_CXX_FLAGS "-std=c++11")
#SET(CMAKE_CXX_FLAGS "-std=c++98")
IF(LINUX)
   SET(CMAKE_CXX_FLAGS "-std=c++17 -fopenmp")
ELSE()
   # find the correct path for isysroot with `xcrun --show-sdk-path`
   SET(CMAKE_CXX_FLAGS "-std=c++17 -Xpreprocessor -fopenmp -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk -I/usr/local/include")
ENDIF()

project(HeatEquation)

  FIND_PACKAGE(ITK)
    IF(ITK_FOUND)
        INCLUDE(${ITK_USE_FILE})
    ELSE(ITK_FOUND)
         MESSAGE(FATAL_ERROR "Cannot build ITKApps without ITK.  Please set ITK_DIR.")
    ENDIF(ITK_FOUND)


# Boost
set(boost_min_ver 1.50.0)
set(boost_libs system filesystem timer chrono)
find_package(Boost ${boost_min_ver})

if(Boost_FOUND)
    find_package(Boost ${boost_min_ver} COMPONENTS ${boost_libs})
endif()

OPTION (USE_OpenMP "Use OpenMP" ON)
IF(USE_OpenMP)
  FIND_PACKAGE(OpenMP)
  IF(OPENMP_FOUND)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  ENDIF()
ENDIF()


add_executable(HeatEquation heat_equation.cxx )
target_link_libraries(HeatEquation ${ITK_LIBRARIES} ${Boost_LIBRARIES} -L/usr/local/Cellar/libomp/9.0.0/lib -lomp)


add_executable(HeatEquationX heat_equation_multiple.cxx )
target_link_libraries(HeatEquationX  ${ITK_LIBRARIES} ${Boost_LIBRARIES})
