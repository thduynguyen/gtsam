# Install headers
set(subdir semiring)
file(GLOB semiring_headers "*.h")
install(FILES ${semiring_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gtsam/semiring)

# Add all tests
add_subdirectory(tests)
