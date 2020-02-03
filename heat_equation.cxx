// example: ./HeatEquation -s 3 -q 3 -n -i 10 data/data_1wm_2ven_3gm.nii /tmp/ -t 4 3 0 2 100.99

// Requires lots of iterations >1000 until convergence
// Would benefit from low-res initialization and successive upsampling strategy
// Would benefit from multi-core/openMP/TBB
// Would benefit from switch between single point floating and double resolution
// Would benefit from implementation as an itk filter such as itk::GradientMagnitudeImageFilter

// compute unit normal and unit bi-normal vector to the unit tangent vector

// we can get faster if we only process the bounding box that contains the changing voxel
// can be calculated once and provided to oneStep as i,j,k,highi,highj,highk

// see http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.136.6443&rep=rep1&type=pdf
// see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3068613/
#include "itkGradientImageFilter.h"
#include "itkGradientMagnitudeImageFilter.h"
#include "itkGradientRecursiveGaussianImageFilter.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"

#include "json.hpp"
#include "metaCommand.h"
#include <boost/chrono.hpp>
#include <boost/chrono/chrono_io.hpp>
#include <boost/filesystem.hpp>
#include <iomanip>
#include <iostream>
#include <map>
#include <omp.h>

#define VERSION_MAJOR 0
#define VERSION_MINOR 0
#define VERSION_PATCH 10

using json = nlohmann::json;
using namespace boost::filesystem;

// save some stats in a result JSON file for provenance
json resultJSON;

// internal storage for the temperature field
std::vector<float> output; // output buffer
std::vector<float> tmpData;
std::vector<int> data; // labels

#define toindex(x, y, z) (size_t)(((z)*dims[0] * dims[1]) + ((y)*dims[0]) + (x))

constexpr unsigned int ImageDimension = 3;
using PixelType = unsigned short;
typedef itk::Image<unsigned short, ImageDimension> ImageType;
typedef itk::ImageFileReader<ImageType> ImageReaderType;

using OutputPixelType = float;
using OutputImageType = itk::Image<OutputPixelType, ImageDimension>;
typedef itk::ImageFileReader<OutputImageType> OutputReaderType;

typedef itk::CovariantVector<OutputPixelType, ImageDimension> GradientPixelType;
typedef itk::Image<GradientPixelType, ImageDimension> GradientImageType;
typedef itk::SmartPointer<GradientImageType> GradientImagePointer;

typedef itk::GradientRecursiveGaussianImageFilter<OutputImageType, GradientImageType> GradientImageFilterType;
typedef typename GradientImageFilterType::Pointer GradientImageFilterPointer;
typedef itk::GradientMagnitudeImageFilter<OutputImageType, OutputImageType> GradientMagnitudeImageFilterType;
typedef typename GradientMagnitudeImageFilterType::Pointer GradientMagnitudeImageFilterPointer;

// compute the magnitude of the gradient field
OutputImageType::Pointer computeMagGradField(OutputImageType::Pointer input) {
  // compute the gradient field
  GradientMagnitudeImageFilterPointer gmfilter = GradientMagnitudeImageFilterType::New();
  gmfilter->SetInput(input);
  // gmfilter->SetSigma(1.0f);
  gmfilter->Update();
  OutputImageType::Pointer gmimage = gmfilter->GetOutput();
  return gmimage;

  // We know what the max and min values for the gradient magnitude are. They are defined
  // by the temperature values we have set in the input.
}

// perform one simulation step, assume that data ends up in output (uses tmpData as temp storage)
double oneStep(ImageType::SizeType dims, std::map<int, float> temperatureByMaterial, unsigned long *boundingBox) {
  float omega = 0.1;
  size_t count = 0;
  bool zeroSpecified = false;
  if (temperatureByMaterial.find(0) != temperatureByMaterial.end()) {
    zeroSpecified = true;
  }
  // omp_set_num_threads(5);
  // printf("  Number of processors available = %d\n", omp_get_num_procs());
  // printf("  Number of threads =              %d\n", omp_get_max_threads());
  // printf("  Number of threads used         = %d\n", omp_get_num_threads());

  size_t ind111, ind101, ind121, ind011, ind211, ind110, ind112;
  float val111, val101, val121, val011, val211, val110, val112;
  float result;
  int i, j, k;
  int im = boundingBox[0];
  int jm = boundingBox[1];
  int km = boundingBox[2];
  int highi = boundingBox[3];
  int highj = boundingBox[4];
  int highk = boundingBox[5];
  if (im < 1)
    im = 1;
  if (jm < 1)
    jm = 1;
  if (km < 1)
    km = 1;
  if (highi > dims[0] - 1)
    highi = dims[0] - 1;
  if (highj > dims[1] - 1)
    highj = dims[1] - 1;
  if (highk > dims[2] - 1)
    highk = dims[2] - 1;

#pragma omp parallel for collapse(3) private(i, j, count, ind111, ind101, ind121, ind011, ind211, ind110, ind112, val111, val101, val121, val011, val211,      \
                                             val110, val112, result) schedule(dynamic, 4)
  for (k = km; k < highk; k++) {
    for (j = jm; j < highj; j++) {
      for (i = im; i < highi; i++) {
        // ok what tissue type is this cell?
        // we only care for either being Exterior or something else
        count = toindex(i, j, k);
        if (data[count] != 0 || zeroSpecified) { // do something (not exterior)
          // what are the values at the stencil around the current location?
          //  001  (101)  201
          // (011) (111) (211)
          //  021  (121)  221
          // and one above and one below that
          // 110, 112
          result = 0.0f;
          if (temperatureByMaterial.find(data[count]) != temperatureByMaterial.end()) { // if the temperature is set, do not change it
            result = temperatureByMaterial[data[count]];
          } else { // otherwise compute the new temperature by the stencil values
            ind111 = count;
            ind101 = toindex(i, j - 1, k);
            ind121 = toindex(i, j + 1, k);
            ind011 = toindex(i - 1, j, k);
            ind211 = toindex(i + 1, j, k);
            ind110 = toindex(i, j, k - 1);
            ind112 = toindex(i, j, k + 1);
            val111 = output[ind111];
            val101 = output[ind101];
            val121 = output[ind121];
            val011 = output[ind011];
            val211 = output[ind211];
            val110 = output[ind110];
            val112 = output[ind112];
            // repulsive borders (to exterior)
            if (data[ind101] == 0)
              val101 = val121;
            if (data[ind121] == 0)
              val121 = val101;
            if (data[ind011] == 0)
              val011 = val211;
            if (data[ind211] == 0)
              val211 = val011;
            if (data[ind110] == 0)
              val110 = val112;
            if (data[ind112] == 0)
              val112 = val110;

            result = (1.0 - 6.0 * omega) * val111 + omega * (val101 + val121 + val011 + val211 + val110 + val112);
            // fprintf(stdout, "label %d = %f\n", data[count], result);
          }
          tmpData[(size_t)count] = result;
        } else {
          tmpData[(size_t)count] = 0.0f;
        }
      }
    }
  }
  // calculate the (absolute) difference between the two fields - todo: use  as convergence criteria
  double diff = 0.0;
  size_t c = output.size() - 1;
  while (c >= 0) {
    diff += fabs(output[c] - tmpData[c]);
    if (c == 0) {
      // underflow of c here!
      break;
    }
    c = c - 1;
  }
  // copy the tmpData to the output
  output = tmpData; // should copy the data after the iteration
  // memcpy(output->lattice.dataPtr(), tmpData.dataPtr(), dims[0] * dims[1] * dims[2] * 4); // float
  // copy
  return diff;
}

std::string beautify_duration(boost::chrono::seconds input_seconds) {
  using namespace boost::chrono;
  typedef duration<int, boost::ratio<86400>> days;
  auto d = duration_cast<days>(input_seconds);
  input_seconds -= d;
  auto h = duration_cast<hours>(input_seconds);
  input_seconds -= h;
  auto m = duration_cast<minutes>(input_seconds);
  input_seconds -= m;
  auto s = duration_cast<seconds>(input_seconds);

  auto dc = d.count();
  auto hc = h.count();
  auto mc = m.count();
  auto sc = s.count();

  std::stringstream ss;
  ss.fill('0');
  if (dc) {
    ss << d.count() << "d";
  }
  if (dc || hc) {
    if (dc) {
      ss << std::setw(2);
    } // pad if second set of numbers
    ss << h.count() << "h";
  }
  if (dc || hc || mc) {
    if (dc || hc) {
      ss << std::setw(2);
    }
    ss << m.count() << "m";
  }
  if (dc || hc || mc || sc) {
    if (dc || hc || mc) {
      ss << std::setw(2);
    }
    ss << s.count() << 's';
  }

  return ss.str();
}

int main(int argc, char *argv[]) {
  itk::MultiThreaderBase::SetGlobalMaximumNumberOfThreads(4);

  std::stringstream version_number;
  version_number << VERSION_MAJOR << "." << VERSION_MINOR << "." << VERSION_PATCH;
  const std::string VERSION_NO = version_number.str();

  MetaCommand command;
  command.SetAuthor("Hauke Bartsch");
  command.SetVersion(VERSION_NO.c_str());
  command.SetDescription("Simulation of the heat equation. Use as in: ./HeatEquation -s 2 -n -q 3 -i 2000 wm.seg.nii /tmp/ "
                         "-t 4 4 0 1 0.99. Specify the -t option at the end.");
  command.SetCategory("MRI");
  command.AddField("infile", "Input mask in nifti or other file format understood by itk", MetaCommand::STRING, true);
  command.AddField("outdir", "Output directory", MetaCommand::STRING, true);

  command.SetOption("Temperatures", "t", false,
                    "Specify the temperature per label as <N> [<label value> <temperature>], with "
                    "N the number of label and temperature values such as in '-t "
                    "4 0 0.0 1 100.0'. A label that is not specified will be "
                    "assumed to be variable and used for the computation (if not label 0).");
  command.SetOptionLongTag("Temperatures", "temperature-label-pairs");
  command.AddOptionField("Temperatures", "temperature", MetaCommand::LIST, true);

  command.SetOption("Iterations", "i", false,
                    "Specify the number of iterations (default 1) the code is run. Suggested is to use a "
                    "large number of iterations like 2000 (see section about speed up). Convergence can be "
                    "monitored using the change value printed for each iteration (sum of absolute differences).");

  command.AddOptionField("Iterations", "iterations", MetaCommand::INT, true);

  // supersample the input (2 means 4 times more voxel)
  command.SetOption("SuperSample", "s", false,
                    "Specify the number up-sampling steps using nearest neighboor interpolation (0 or 1 have no effect, 2 doubles the resolution 0.5 half's "
                    "the resolution).");
  command.AddOptionField("SuperSample", "supersample", MetaCommand::FLOAT, true);

  // quantize the output temperature
  command.SetOption("Quantize", "q", false, "Quantize the output into N different regions of equal volume.");
  command.AddOptionField("Quantize", "quantize", MetaCommand::INT, true);

  command.SetOption("UnitNormalVector", "n", false,
                    "Export the unit normal vector and the unit binormal vector per voxel "
                    "(exported gradient field is the tangent vector) in nrrd format or in another specified vector format (see -f).");

  command.SetOption(
      "InitField", "c", false,
      "Initialize the temperature field with this volume. This option together with the super sample option can be used to speed up convergence if a sequence "
      "of small to large volumes is created where each stage is initialized with the temperature field calculated from the previous stage.");
  command.AddOptionField("InitField", "initfield", MetaCommand::STRING, true);

  command.SetOption("VectorFileFormat", "f", false,
                    "Specify the file format for the vector fields. Can be -f nrrd or -f vtk or another format supported by itk.");
  command.AddOptionField("VectorFileFormat", "vectorfileformat", MetaCommand::STRING, true);

  if (!command.Parse(argc, argv)) {
    return 1;
  }

  std::string input = command.GetValueAsString("infile");
  std::string outdir = command.GetValueAsString("outdir");
  // fprintf(stdout, "input: \"%s\"\n", input.c_str());
  // fprintf(stdout, "outdir: \"%s\"\n", outdir.c_str());
  if (!boost::filesystem::exists(input)) {
    std::cout << "Could not find the input file " << input << "..." << std::endl;
    exit(1);
  }

  std::string initfield;
  bool useInitField = false;
  if (command.GetOptionWasSet("InitField")) {
    initfield = command.GetValueAsString("InitField", "initfield");
    useInitField = true;
  }

  std::string vectorfileextension = "nrrd";
  if (command.GetOptionWasSet("VectorFileFormat")) {
    if (command.GetValueAsString("VectorFileFormat", "vectorfileformat") == "nii") {
      fprintf(stdout, "Warning: No nii support for vector fields, will use nrrd instead.\n");
    } else {
      vectorfileextension = command.GetValueAsString("VectorFileFormat", "vectorfileformat");
    }
  }

  float supersampling = 0;
  if (command.GetOptionWasSet("SuperSample"))
    supersampling = command.GetValueAsFloat("SuperSample", "supersample");
  if (supersampling < 0) {
    fprintf(stdout, "Error: don't know how to supersample with negative values...\n");
    exit(-1);
  }
  int quantize = -1; // don't quantize
  if (command.GetOptionWasSet("Quantize"))
    quantize = command.GetValueAsInt("Quantize", "quantize");

  // todo: instead of number of iterations it would be good to have convergence error (might require
  // double computations)
  int iterations = 1;
  if (command.GetOptionWasSet("Iterations"))
    iterations = command.GetValueAsInt("Iterations", "iterations");

  // computed using finite differences and cross-product
  bool unitNormal = false;
  if (command.GetOptionWasSet("UnitNormalVector"))
    unitNormal = true;

  std::map<int, float> temperatureByMaterial;

  std::vector<int> labels;
  std::vector<float> temperatures;

  std::string temp_str = ""; // > 0
  if (command.GetOptionWasSet("Temperatures")) {
    std::list<std::string> thresholds = command.GetValueAsList("Temperatures");
    std::list<std::string>::iterator it;
    if (thresholds.size() % 2 != 0) {
      fprintf(stdout,
              "Error: should be an even number of threshold values and temperatures but found %lu "
              "entries.\n",
              thresholds.size());
      exit(-1);
    }
    // fprintf(stdout, "found %lu temperature arguments\n", thresholds.size());
    resultJSON["temperatures"] = json::array();
    // std::list<std::string>::iterator it;
    for (it = thresholds.begin(); it != thresholds.end(); it++) {
      // append to labels and temperatures
      int mat = atoi((*it).c_str());
      float temp = 0.0;
      if (it != thresholds.end()) {
        it++;
        temp = atof((*it).c_str());
      }
      temperatureByMaterial.insert(std::make_pair(mat, temp));
      json v;
      v["label"] = mat;
      v["temperature"] = temp;
      resultJSON["temperatures"].push_back(v);
    }
  }
  bool verbose = false;
  if (command.GetOptionWasSet("Verbose"))
    verbose = true;

  // store information in the result json file
  resultJSON["command_line"] = json::array();
  for (int i = 0; i < argc; i++) {
    resultJSON["command_line"].push_back(std::string(argv[i]));
  }
  path p(input);
  std::string fn = p.filename().string();
  size_t lastdot = fn.find_last_of(".");
  std::string output_filename;
  if (lastdot == std::string::npos)
    output_filename = fn + "_temperature.nii";
  else
    output_filename = fn.substr(0, lastdot) + "_temperature.nii";

  std::string output_filename2;
  if (lastdot == std::string::npos)
    output_filename2 = fn + "_gradient." + vectorfileextension;
  else
    output_filename2 = fn.substr(0, lastdot) + "_gradient." + vectorfileextension;

  std::string output_filename3;
  if (lastdot == std::string::npos)
    output_filename3 = fn + "_gradient_normal." + vectorfileextension;
  else
    output_filename3 = fn.substr(0, lastdot) + "_gradient_normal." + vectorfileextension;

  std::string output_filename4;
  if (lastdot == std::string::npos)
    output_filename4 = fn + "_gradient_binormal." + vectorfileextension;
  else
    output_filename4 = fn.substr(0, lastdot) + "_gradient_binormal." + vectorfileextension;

  resultJSON["output_temperature"] = outdir + "/" + output_filename;
  resultJSON["output_gradient"] = outdir + "/" + output_filename2;

  ImageReaderType::Pointer imageReader = ImageReaderType::New();
  imageReader->SetFileName(input);
  imageReader->Update();

  ImageType::Pointer inputVol = imageReader->GetOutput();
  ImageType::SpacingType spacing = inputVol->GetSpacing();
  ImageType::RegionType region = inputVol->GetLargestPossibleRegion();
  ImageType::SizeType dims = region.GetSize();
  ImageType::PointType origin = inputVol->GetOrigin();

  // do supersampling if required - can be support sub-sampling as well?
  // that would make it easy to implement a staged computation across an image pyramid
  if (supersampling > 0) {
    resultJSON["SuperSamplingFactor"] = supersampling;
    using ResampleFilterType = itk::ResampleImageFilter<ImageType, ImageType>;

    ResampleFilterType::Pointer resampler = ResampleFilterType::New();
    using TransformType = itk::IdentityTransform<double, 3>;

    TransformType::Pointer transform = TransformType::New();
    transform->SetIdentity();

    resampler->SetTransform(transform);
    // using InterpolatorType = itk::LinearInterpolateImageFunction<ImageType, double>;
    using InterpolatorType = itk::NearestNeighborInterpolateImageFunction<ImageType, double>;

    InterpolatorType::Pointer interpolator = InterpolatorType::New();

    resampler->SetInterpolator(interpolator);

    resampler->SetDefaultPixelValue(0); // highlight regions without source

    OutputImageType::SpacingType spacingOut;

    spacingOut[0] = spacing[0] / supersampling;
    spacingOut[1] = spacing[1] / supersampling;
    spacingOut[2] = spacing[2] / supersampling;
    resultJSON["OutputSpacing"] = json::array();
    resultJSON["OutputSpacing"].push_back(spacingOut[0]);
    resultJSON["OutputSpacing"].push_back(spacingOut[1]);
    resultJSON["OutputSpacing"].push_back(spacingOut[2]);

    resampler->SetOutputSpacing(spacingOut);
    resampler->SetOutputOrigin(inputVol->GetOrigin());
    resampler->SetOutputDirection(inputVol->GetDirection());
    ImageType::SizeType size;

    size[0] = static_cast<itk::SizeValueType>(dims[0] * supersampling);
    size[1] = static_cast<itk::SizeValueType>(dims[1] * supersampling);
    size[2] = static_cast<itk::SizeValueType>(dims[2] * supersampling);
    resultJSON["OutputSize"] = json::array();
    resultJSON["OutputSize"].push_back(size[0]);
    resultJSON["OutputSize"].push_back(size[1]);
    resultJSON["OutputSize"].push_back(size[2]);

    resampler->SetSize(size);
    resampler->SetInput(inputVol);

    resampler->Update();
    inputVol = resampler->GetOutput();
  }
  spacing = inputVol->GetSpacing();
  region = inputVol->GetLargestPossibleRegion();
  dims = region.GetSize();
  origin = inputVol->GetOrigin();

  // compute the bounding box and the dimensions
  float bb[6];
  bb[0] = origin[0];
  bb[2] = origin[1];
  bb[4] = origin[2];
  bb[1] = bb[0] + spacing[0] * (dims[0] - 1);
  bb[3] = bb[2] + spacing[1] * (dims[1] - 1);
  bb[5] = bb[4] + spacing[2] * (dims[2] - 1);
  // fprintf(stdout, "BoundingBox: %f %f %f %f %f %f\n", bb[0], bb[1], bb[2], bb[3], bb[4], bb[5]);

  // copy the data into the data and tmpData buffers
  output.resize(dims[0] * dims[1] * dims[2]); // the output temperature as float
  if (useInitField) {
    fprintf(stdout, "copy initial temperature values from init field using resampling...\n");
    // regardless of the resolution of the input file we need to resample it to the output file (and copy to output)
    OutputReaderType::Pointer initReader = OutputReaderType::New();
    initReader->SetFileName(initfield);
    initReader->Update();

    // after supersampling inputVol has the resolution of the output we need
    ImageType::SpacingType spacing = inputVol->GetSpacing();
    ImageType::RegionType region = inputVol->GetLargestPossibleRegion();
    ImageType::SizeType dims = region.GetSize();
    ImageType::PointType origin = inputVol->GetOrigin();

    using OutputResampleFilterType = itk::ResampleImageFilter<OutputImageType, OutputImageType>;

    OutputResampleFilterType::Pointer resampler2 = OutputResampleFilterType::New();
    using TransformType = itk::IdentityTransform<double, 3>;

    // keep the same transformation as the input
    TransformType::Pointer transform = TransformType::New();
    transform->SetIdentity();

    resampler2->SetTransform(transform);
    // we should use a better  interpolator here (cubic)
    using InterpolatorType = itk::LinearInterpolateImageFunction<OutputImageType, double>;
    // using InterpolatorType = itk::NearestNeighborInterpolateImageFunction<ImageType, double>;

    InterpolatorType::Pointer interpolator = InterpolatorType::New();
    resampler2->SetInterpolator(interpolator);
    resampler2->SetDefaultPixelValue(-10); // highlight regions without source

    resampler2->SetOutputSpacing(spacing);
    resampler2->SetOutputOrigin(inputVol->GetOrigin());
    resampler2->SetOutputDirection(inputVol->GetDirection());

    resampler2->SetSize(dims);
    resampler2->SetInput(initReader->GetOutput());

    resampler2->Update();
    // now copy to data to the output array and use it during the iterations
    OutputImageType::Pointer initVol = resampler2->GetOutput();
    OutputImageType::RegionType initRegion = initVol->GetLargestPossibleRegion();
    itk::ImageRegionIterator<OutputImageType> initIterator(initVol, initRegion);
    // todo: we should make sure that the fixed temperature regions have the correct initial values
    while (!initIterator.IsAtEnd()) {
      OutputImageType::IndexType pixelIndex = initIterator.GetIndex();
      size_t counter = toindex(pixelIndex[0], pixelIndex[1], pixelIndex[2]); // slow but correct
      if (counter > 0 && counter < output.size() - 1) {
        output[counter] = initIterator.Get();
      }
      ++initIterator;
    }
    // we should free the temporary resampled volume here again - hopefully that happens on its own after this block
  } else {
    std::fill(output.begin(), output.end(), 0.0);
  }

  unsigned long boundingBox[6] = {dims[0], dims[1], dims[2], 0, 0, 0}; // i,j,k,highi,highj,highk
  data.resize(dims[0] * dims[1] * dims[2]);                            // the labels as int
  itk::ImageRegionIterator<ImageType> volIterator(inputVol, region);
  size_t counter = 0;
  while (!volIterator.IsAtEnd()) {
    data[counter] = volIterator.Get();
    ImageType::IndexType pixelIndex = volIterator.GetIndex();
    if (data[counter] != 0 && temperatureByMaterial.find(data[counter]) == temperatureByMaterial.end()) {
      // at this index we would need to compute a value
      if (boundingBox[0] > pixelIndex[0])
        boundingBox[0] = pixelIndex[0];
      if (boundingBox[1] > pixelIndex[1])
        boundingBox[1] = pixelIndex[1];
      if (boundingBox[2] > pixelIndex[2])
        boundingBox[2] = pixelIndex[2];
      if (boundingBox[3] < pixelIndex[0])
        boundingBox[3] = pixelIndex[0];
      if (boundingBox[4] < pixelIndex[1])
        boundingBox[4] = pixelIndex[1];
      if (boundingBox[5] < pixelIndex[2])
        boundingBox[5] = pixelIndex[2];
    }
    counter++;
    ++volIterator;
  }
  tmpData.resize(dims[0] * dims[1] * dims[2]); // temporary temperatures as float
  std::fill(tmpData.begin(), tmpData.end(), 0.0);
  fprintf(stdout, "bounding box limited computation %lu, %lu, %lu .. %lu, %lu, %lu\n", boundingBox[0], boundingBox[1], boundingBox[2], boundingBox[3],
          boundingBox[4], boundingBox[5]);

  // run the iterations
  boost::chrono::system_clock::time_point start = boost::chrono::system_clock::now();
  for (int i = 0; i < iterations; i++) {
    // single iteration
    // boost::chrono::system_clock::time_point begin = boost::chrono::system_clock::now();
    fprintf(stdout, "step: %d/%d", i + 1, iterations);
    double change = oneStep(dims, temperatureByMaterial, boundingBox);
    // boost::chrono::system_clock::time_point end = boost::chrono::system_clock::now();
    // float howmanymoreseconds = boost::chrono::duration_cast<boost::chrono::seconds>(end - begin).count() * (iterations - (i + 1));

    boost::chrono::duration<double> sec = boost::chrono::system_clock::now() - start;
    float predicted = sec.count() / (float)i; // time per iteration
    fprintf(stdout, " change: %g (%.2f, done in: %.2f)\n", change, predicted * i, predicted * (iterations - i));

    if (change == 0) {
      break; // early stopping
    }
  }

  // create the output object and save
  OutputImageType::Pointer outVol = OutputImageType::New();
  outVol->SetRegions(region);
  outVol->Allocate();
  outVol->SetOrigin(inputVol->GetOrigin());
  outVol->SetSpacing(inputVol->GetSpacing());
  outVol->SetDirection(inputVol->GetDirection());
  itk::ImageRegionIterator<OutputImageType> outIterator(outVol, region);
  counter = 0;
  while (!outIterator.IsAtEnd()) {
    outIterator.Set(output[counter]);
    counter++;
    ++outIterator;
  }

  // export the potential field
  typedef itk::ImageFileWriter<OutputImageType> WriterType;
  WriterType::Pointer writer = WriterType::New();
  // check if that directory exists, create before writing
  writer->SetFileName(resultJSON["output_temperature"]);
  writer->SetInput(outVol);
  // check if that directory exists, create before writing
  std::string p_txt = resultJSON["output_temperature"];
  path out_p(p_txt);
  create_directories(out_p.parent_path());

  std::cout << "Writing the temperature field as ";
  std::cout << resultJSON["output_temperature"] << std::endl;

  try {
    writer->Update();
  } catch (itk::ExceptionObject &ex) {
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }

  // compute the gradient field
  GradientImageFilterPointer gfilter = GradientImageFilterType::New();
  gfilter->SetInput(outVol);
  gfilter->SetSigma(1.0f);
  gfilter->Update();
  GradientImagePointer gimage = gfilter->GetOutput();
  GradientImagePointer gimage_for_export = gimage;

  // mask the gradient field using the fixed temperature values
  if (1) { // mask the gradient field, set all vectors to 0 that have a fixed temperature
    GradientImageType::Pointer gimage_masked = GradientImageType::New();
    gimage_masked->SetRegions(region);
    gimage_masked->Allocate();
    gimage_masked->SetOrigin(inputVol->GetOrigin());
    gimage_masked->SetSpacing(inputVol->GetSpacing());
    gimage_masked->SetDirection(inputVol->GetDirection());
    itk::ImageRegionIterator<GradientImageType> oIterator(gimage_masked, region);
    itk::ImageRegionIterator<GradientImageType> iIterator(gimage, region);
    itk::ImageRegionIterator<ImageType> maskIterator(inputVol, region);
    using VectorType = itk::CovariantVector<double, 3>;

    while (!oIterator.IsAtEnd() && !maskIterator.IsAtEnd() && !iIterator.IsAtEnd()) {
      if (temperatureByMaterial.find(maskIterator.Get()) != temperatureByMaterial.end()) {
        // this is a voxel with a fixed temperature, set gradient to 0
        VectorType erg;
        erg[0] = 0;
        erg[1] = 0;
        erg[2] = 0;
        oIterator.Set(erg); // set to 0
      } else {
        oIterator.Set(iIterator.Get()); // set to input
      }
      ++oIterator;
      ++iIterator;
      ++maskIterator;
    }
    gimage_for_export = gimage_masked;
  }

  // export the gradient of the potential field
  typedef itk::ImageFileWriter<GradientImageType> GradientWriterType;
  GradientWriterType::Pointer writer2 = GradientWriterType::New();
  // check if that directory exists, create before writing
  writer2->SetFileName(resultJSON["output_gradient"]); // this is the tangent unit vector
  writer2->SetInput(gimage_for_export);

  std::cout << "Writing the gradient of the temperature field as ";
  std::cout << resultJSON["output_gradient"] << std::endl;

  try {
    writer2->Update();
  } catch (itk::ExceptionObject &ex) {
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }

  // quantize the output temperature
  if (quantize > 0) {
    std::string output_filename3;
    if (lastdot == std::string::npos)
      output_filename3 = fn + "_temperature_quantized.nii";
    else
      output_filename3 = fn.substr(0, lastdot) + "_temperature_quantized.nii";

    resultJSON["output_temperature_quantized"] = outdir + "/" + output_filename3;

    // what is the temperature range we need to quantize?

    OutputImageType::Pointer outQuant = OutputImageType::New();
    outQuant->SetRegions(region);
    outQuant->Allocate();
    outQuant->SetOrigin(inputVol->GetOrigin());
    outQuant->SetSpacing(inputVol->GetSpacing());
    outQuant->SetDirection(inputVol->GetDirection());
    itk::ImageRegionIterator<OutputImageType> oIterator(outQuant, region);
    // compute the quartiles for all voxel in the non-zero non-fixed temperature voxel
    int h_size = 200;
    std::vector<size_t> histogram(h_size);
    std::map<int, float>::iterator mit;
    float maxTemp, minTemp; // histogram maps from max to min temperature
    for (mit = temperatureByMaterial.begin(); mit != temperatureByMaterial.end(); mit++) {
      if (mit == temperatureByMaterial.begin()) {
        maxTemp = minTemp = mit->second;
      }
      if (mit->second > maxTemp)
        maxTemp = mit->second;
      if (mit->second < minTemp)
        minTemp = mit->second;
    }
    resultJSON["temperature_range_specified"] = json::array();
    resultJSON["temperature_range_specified"].push_back(minTemp);
    resultJSON["temperature_range_specified"].push_back(maxTemp);

    itk::ImageRegionIterator<OutputImageType> temperatureIterator(outVol, region);
    itk::ImageRegionIterator<ImageType> maskIterator(inputVol, region);

    while (!temperatureIterator.IsAtEnd() && !maskIterator.IsAtEnd()) {
      if (maskIterator.Get() != 0) {
        if (temperatureByMaterial.find(maskIterator.Get()) == temperatureByMaterial.end()) {
          // this label does not have a fixed temperature, lets use it
          int idx = ((temperatureIterator.Get() - minTemp) / (maxTemp - minTemp)) * (h_size - 1);
          if (idx < 0)
            idx = 0;
          if (idx > h_size - 1)
            idx = h_size - 1;

          histogram[idx]++;
        }
      }
      ++temperatureIterator;
      ++maskIterator;
    }
    // now compute the normalized cummulative histogram
    std::vector<double> cum_hist(h_size);
    double sum = 0.0;
    for (int i = 0; i < h_size; i++) {
      sum += histogram[i];
      cum_hist[i] = histogram[i];
      if (i > 0) {
        cum_hist[i] = cum_hist[i] + cum_hist[i - 1];
      }
    }
    for (int i = 0; i < h_size; i++) {
      cum_hist[i] /= sum;
    }
    std::vector<double> quartiles(quantize - 1); // we got one less border than we have quantiles
    // now set the threshold temperature for each quartile range
    for (int i = 0; i < quantize - 1; i++) {
      // what is the first temperature where we reach the current quantile?
      double quant_step = (i + 1) * (1.0 / quantize); // lower border of quantile
      for (int j = 0; j < h_size; j++) {
        if (cum_hist[j] >= quant_step) {
          quartiles[i] = ((float)j / (h_size - 1.0)) * (maxTemp - minTemp) + minTemp; // temperature at this index
          break;
        }
      }
    }
    json ar = json::array();
    for (int i = 0; i < quartiles.size(); i++) {
      ar.push_back(quartiles[i]);
    }
    resultJSON["output_temperature_quantized_thresholds"] = ar;

    temperatureIterator.GoToBegin();
    maskIterator.GoToBegin();
    // itk::ImageRegionIterator<ImageType> maskIterator(inputVol, region);
    itk::ImageRegionIterator<OutputImageType> outputIterator(outQuant, region);
    while (!temperatureIterator.IsAtEnd() && !maskIterator.IsAtEnd() && !outputIterator.IsAtEnd()) {
      outputIterator.Set(0); // outside
      if (maskIterator.Get() != 0) {
        if (temperatureByMaterial.find(maskIterator.Get()) == temperatureByMaterial.end()) {
          // this label does not have a fixed temperature, lets use it
          // what is the quantile for this voxel?
          int q = 1;
          for (int i = quartiles.size() - 1; i >= 0; i--) {
            if (temperatureIterator.Get() > quartiles[i]) {
              q = i + 2; // start counting from 1
              break;
            }
          }
          outputIterator.Set(q);
        }
      }
      ++temperatureIterator;
      ++maskIterator;
      ++outputIterator;
    }

    // export the quantized temperature field
    typedef itk::ImageFileWriter<OutputImageType> WriterType;
    WriterType::Pointer writer = WriterType::New();
    // check if that directory exists, create before writing
    writer->SetFileName(resultJSON["output_temperature_quantized"]);
    writer->SetInput(outQuant);

    std::cout << "Writing the temperature field as ";
    std::cout << resultJSON["output_temperature_quantized"] << std::endl;

    try {
      writer->Update();
    } catch (itk::ExceptionObject &ex) {
      std::cout << ex << std::endl;
      return EXIT_FAILURE;
    }
  }

  // Unit Normal Vector calculation (and computation of the unit binormal)
  if (unitNormal) {
    resultJSON["output_gradient_normal"] = outdir + "/" + output_filename3;
    resultJSON["output_gradient_binormal"] = outdir + "/" + output_filename4;

    fprintf(stdout, "compute unit normal vector direction for each voxel using finite differences...\n");
    // the tangent vector is in gimage, walk through the different voxel in x, y, z and compute
    // finite differences GradientPixelType as location at each point
    GradientImageType::Pointer outUN = GradientImageType::New();
    outUN->SetRegions(region);
    outUN->Allocate();
    outUN->SetOrigin(inputVol->GetOrigin());
    outUN->SetSpacing(inputVol->GetSpacing());
    outUN->SetDirection(inputVol->GetDirection());

    GradientImageType::Pointer outUBN = GradientImageType::New();
    outUBN->SetRegions(region);
    outUBN->Allocate();
    outUBN->SetOrigin(inputVol->GetOrigin());
    outUBN->SetSpacing(inputVol->GetSpacing());
    outUBN->SetDirection(inputVol->GetDirection());

    itk::ImageRegionIterator<GradientImageType> tangentIterator(gimage, region);
    itk::ImageRegionIterator<GradientImageType> unIterator(outUN, region);
    itk::ImageRegionIterator<GradientImageType> ubnIterator(outUBN, region);
    itk::ImageRegionIterator<ImageType> maskIterator(inputVol, region);
    using PointType = itk::Point<GradientPixelType, 3>;
    using VectorType = itk::CovariantVector<double, 3>;
    GradientImageType::IndexType pixelIndex;
    using PointType = itk::Point<GradientPixelType, 3>;
    ImageType::IndexType idxPoint1;
    ImageType::IndexType idxPoint2;
    double dsx = inputVol->GetSpacing()[0] * 2; // we move over the middle pixel so dT/ds
    double dsy = inputVol->GetSpacing()[1] * 2;
    double dsz = inputVol->GetSpacing()[2] * 2;
    while (!tangentIterator.IsAtEnd() && !maskIterator.IsAtEnd() && !unIterator.IsAtEnd() && !ubnIterator.IsAtEnd()) {
      GradientPixelType p = tangentIterator.Get();
      std::vector<float> vec(3);
      ImageType::IndexType pixelIndex = tangentIterator.GetIndex();
      VectorType point0 = gimage->GetPixel(pixelIndex); // pull the data at this pixel location
      //
      // x - direction
      //
      double ds = dsx;
      if (pixelIndex[0] - 1 < 0) {
        // substitute center pixel
        idxPoint1 = {{pixelIndex[0], pixelIndex[1], pixelIndex[2]}};
        ds /= 2.0; // only half the distance
      } else {
        idxPoint1 = {{pixelIndex[0] - 1, pixelIndex[1], pixelIndex[2]}};
      }
      if (pixelIndex[0] + 1 >= dims[0]) {
        // substitute center pixel
        idxPoint2 = {{pixelIndex[0], pixelIndex[1], pixelIndex[2]}};
      } else {
        idxPoint2 = {{pixelIndex[0] + 1, pixelIndex[1], pixelIndex[2]}};
      }
      // compute the magnitude of the difference at these locations
      VectorType point1 = gimage->GetPixel(idxPoint1); // pull the data at this pixel location
      VectorType point2 = gimage->GetPixel(idxPoint2); // pull the data at this pixel location
      vec[0] = (point1[0] - point2[0]) * (point1[0] - point2[0]) + (point1[1] - point2[1]) * (point1[1] - point2[1]) +
               (point1[2] - point2[2]) * (point1[2] - point2[2]);
      vec[0] /= ds;
      //
      // y - direction
      //
      ds = dsy;
      if (pixelIndex[1] - 1 < 0) {
        // substitute center pixel
        idxPoint1 = {{pixelIndex[0], pixelIndex[1], pixelIndex[2]}};
        ds /= 2.0;
      } else {
        idxPoint1 = {{pixelIndex[0], pixelIndex[1] - 1, pixelIndex[2]}};
      }
      if (pixelIndex[1] + 1 >= dims[1]) {
        // substitute center pixel
        idxPoint2 = {{pixelIndex[0], pixelIndex[1], pixelIndex[2]}};
      } else {
        idxPoint2 = {{pixelIndex[0], pixelIndex[1] + 1, pixelIndex[2]}};
      }
      // compute the magnitude of the difference at these locations
      point1 = gimage->GetPixel(idxPoint1); // pull the data at this pixel location
      point2 = gimage->GetPixel(idxPoint2); // pull the data at this pixel location
      vec[1] = (point1[0] - point2[0]) * (point1[0] - point2[0]) + (point1[1] - point2[1]) * (point1[1] - point2[1]) +
               (point1[2] - point2[2]) * (point1[2] - point2[2]);
      vec[1] /= ds;
      //
      // z - direction
      //
      ds = dsz;
      if (pixelIndex[2] - 1 < 0) {
        // substitute center pixel
        idxPoint1 = {{pixelIndex[0], pixelIndex[1], pixelIndex[2]}};
        ds /= 2.0;
      } else {
        idxPoint1 = {{pixelIndex[0], pixelIndex[1], pixelIndex[2] - 1}};
      }
      if (pixelIndex[2] + 1 >= dims[2]) {
        // substitute center pixel
        idxPoint2 = {{pixelIndex[0], pixelIndex[1], pixelIndex[2]}};
      } else {
        idxPoint2 = {{pixelIndex[0], pixelIndex[1], pixelIndex[2] + 1}};
      }
      // compute the magnitude of the difference at these locations
      point1 = gimage->GetPixel(idxPoint1); // pull the data at this pixel location
      point2 = gimage->GetPixel(idxPoint2); // pull the data at this pixel location
      vec[2] = (point1[0] - point2[0]) * (point1[0] - point2[0]) + (point1[1] - point2[1]) * (point1[1] - point2[1]) +
               (point1[2] - point2[2]) * (point1[2] - point2[2]);
      vec[2] /= ds;
      // fprintf(stdout, "%f %f %f\n", vec[0], vec[1], vec[2]);
      // scale the result vector to length 1
      double vec_size = sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
      if (vec_size > 0) {
        vec[0] /= vec_size;
        vec[1] /= vec_size;
        vec[2] /= vec_size;
      }
      VectorType erg;
      erg[0] = vec[0];
      erg[1] = vec[1];
      erg[2] = vec[2];
      unIterator.Set(erg);

      std::vector<double> vec2(3); // compute the binormal vector as the cross-product
      vec2[0] = vec[1] * point0[2] - vec[2] * point0[1];
      vec2[1] = vec[2] * point0[0] - vec[0] * point0[2];
      vec2[2] = vec[0] * point0[1] - vec[1] * point0[0];

      erg[0] = vec2[0];
      erg[1] = vec2[1];
      erg[2] = vec2[2];
      ubnIterator.Set(erg);

      ++unIterator;
      ++ubnIterator;
      ++tangentIterator;
      ++maskIterator;
    }

    typedef itk::ImageFileWriter<GradientImageType> GradientWriterType;
    GradientWriterType::Pointer writer3 = GradientWriterType::New();
    // check if that directory exists, create before writing
    writer3->SetFileName(resultJSON["output_gradient_normal"]);
    writer3->SetInput(outUN);

    std::cout << "Writing the unit normal of the gradient of the temperature field as ";
    std::cout << resultJSON["output_gradient_normal"] << std::endl;

    try {
      writer3->Update();
    } catch (itk::ExceptionObject &ex) {
      std::cout << ex << std::endl;
      return EXIT_FAILURE;
    }

    GradientWriterType::Pointer writer4 = GradientWriterType::New();
    // check if that directory exists, create before writing
    writer4->SetFileName(resultJSON["output_gradient_binormal"]);
    writer4->SetInput(outUBN);

    std::cout << "Writing the unit binormal of the gradient of the temperature field as ";
    std::cout << resultJSON["output_gradient_binormal"] << std::endl;

    try {
      writer4->Update();
    } catch (itk::ExceptionObject &ex) {
      std::cout << ex << std::endl;
      return EXIT_FAILURE;
    }
  }

  std::ostringstream o;
  std::string si(outdir + "/" + output_filename);
  si.erase(std::remove(si.begin(), si.end(), '\"'), si.end());
  lastdot = si.find_last_of(".");
  if (lastdot == std::string::npos)
    si = si + ".json";
  else
    si = si.substr(0, lastdot) + ".json";

  o << si;
  std::ofstream out(o.str());
  std::string res = resultJSON.dump(4) + "\n";
  out << res;
  out.close();

  fprintf(stdout, "%s", res.c_str());
}
