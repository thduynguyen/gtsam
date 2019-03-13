# Install cython components
include(GtsamPybindWrap)

set(WRAP_BOOST_LIBRARIES
  Boost::system
  Boost::filesystem
  Boost::thread
)

# Allow for disabling serialization to handle errors related to Clang's linker
option(GTSAM_WRAP_SERIALIZATION "If enabled, allows for wrapped objects to be saved via boost.serialization" ON)

# Build the executable itself
file(GLOB wrap_srcs "*.cpp")
file(GLOB wrap_headers "*.h")
list(REMOVE_ITEM wrap_srcs ${CMAKE_CURRENT_SOURCE_DIR}/wrap.cpp)
add_library(wrap_lib STATIC ${wrap_srcs} ${wrap_headers})
target_include_directories(wrap_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)
if (NOT GTSAM_WRAP_SERIALIZATION)
  target_compile_definitions(wrap_lib PUBLIC -DWRAP_DISABLE_SERIALIZE)
endif()

