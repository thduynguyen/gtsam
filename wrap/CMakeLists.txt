include(GtsamPybindWrap)

# Create the python toolbox for the gtsam library
if(GTSAM_INSTALL_PYTHON_TOOLBOX)
  # wrap gtsam add_custom_target(gtsam_header DEPENDS "../gtsam.h")
  set(ignore
      gtsam::Point2
      gtsam::Point3
      gtsam::LieVector
      gtsam::LieMatrix
      gtsam::ISAM2ThresholdMapValue
      gtsam::FactorIndices)
  pybind_wrapper(
    gtsam_py # target
    ${CMAKE_CURRENT_SOURCE_DIR}/../gtsam.h # interface_header
    "gtsam_py.cpp" # generated_cpp
    "gtsam_py" # module_name
    "" # top_namespace ignore_classes
    "${ignore}"
    gtsam # libs
    "gtsam;gtsam_header") # dependencies
endif()

target_link_libraries(wrap_lib ${WRAP_BOOST_LIBRARIES})
gtsam_assign_source_folders(${wrap_srcs} ${wrap_headers})
add_executable(wrap wrap.cpp)
target_link_libraries(wrap PRIVATE wrap_lib)

# Set folder in Visual Studio
file(RELATIVE_PATH relative_path "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(wrap_lib wrap PROPERTIES FOLDER "${relative_path}")

# Install wrap binary and export target
install(TARGETS wrap EXPORT GTSAM-exports DESTINATION bin)
list(APPEND GTSAM_EXPORTED_TARGETS wrap)
set(GTSAM_EXPORTED_TARGETS "${GTSAM_EXPORTED_TARGETS}" PARENT_SCOPE)

# Install matlab header
install(FILES matlab.h DESTINATION include/wrap)

# Build tests
add_subdirectory(tests)
