// example: ./HeatEquation -i 10 data/data_1wm_2ven_3gm.nii /tmp/ -t 4 3 0 2 100.99

// requires lots of iterations >1000
// would benefit from low-res initialization and upsampling
// would benefit from multi-core

#include "itkGradientImageFilter.h"
#include "itkGradientRecursiveGaussianImageFilter.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"

#include "json.hpp"
#include "metaCommand.h"
#include <boost/filesystem.hpp>
#include <map>

using json = nlohmann::json;
using namespace boost::filesystem;

json resultJSON;

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

typedef itk::CovariantVector<OutputPixelType, ImageDimension> GradientPixelType;
typedef itk::Image<GradientPixelType, ImageDimension> GradientImageType;
typedef itk::SmartPointer<GradientImageType> GradientImagePointer;

typedef itk::GradientRecursiveGaussianImageFilter<OutputImageType, GradientImageType> GradientImageFilterType;
typedef typename GradientImageFilterType::Pointer GradientImageFilterPointer;

// perform one simulation step, assume that data ends up in data (uses tmpData as temp storage)
void oneStep(ImageType::SizeType dims, std::map<int, float> temperatureByMaterial) {
  float omega = 0.1;
  size_t count = 0;
  for (int k = 1; k < dims[2] - 1; k++) {
    for (int j = 1; j < dims[1] - 1; j++) {
      for (int i = 1; i < dims[0] - 1; i++) {
        // ok what tissue type is this cell?
        // we only care for either being Exterior or something else
        count = toindex(i, j, k);
        if (data[count] != 0) { // do something (not exterior)
          // what are the values at the stencil around the current location?
          //  001  (101)  201
          // (011) (111) (211)
          //  021  (121)  221
          // and one above and one below that
          // 110, 112
          size_t ind111 = count;
          size_t ind101 = toindex(i, j - 1, k);
          size_t ind121 = toindex(i, j + 1, k);
          size_t ind011 = toindex(i - 1, j, k);
          size_t ind211 = toindex(i + 1, j, k);
          size_t ind110 = toindex(i, j, k - 1);
          size_t ind112 = toindex(i, j, k + 1);
          float result = 0.0f;
          if (temperatureByMaterial.find(data[count]) != temperatureByMaterial.end()) { // if the temperature is set, do not change it
            result = temperatureByMaterial[data[count]];
          } else { // otherwise compute the new temperature by the stencil values
            float val111 = output[ind111];
            float val101 = output[ind101];
            float val121 = output[ind121];
            float val011 = output[ind011];
            float val211 = output[ind211];
            float val110 = output[ind110];
            float val112 = output[ind112];
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
  // copy the tmpData to the output
  output = tmpData; // should copy the data after the iteration
  // memcpy(output->lattice.dataPtr(), tmpData.dataPtr(), dims[0] * dims[1] * dims[2] * 4); // float copy
}

int main(int argc, char *argv[]) {

  itk::MultiThreaderBase::SetGlobalMaximumNumberOfThreads(4);

  MetaCommand command;
  command.SetAuthor("Hauke Bartsch");
  command.SetDescription("Simulation of the heat equation. Use as in: ./HeatEquation -i 1000 wm.seg.nii /tmp/ -t 4 4 0 1 0.99");
  command.AddField("infile", "Input mask", MetaCommand::STRING, true);
  command.AddField("outdir", "Output mask directory", MetaCommand::STRING, true);

  command.SetOption(
      "Temperatures", "t", false,
      "Specify the temperature for each label as <N> [<label value> <temperature>] such as in '-t 2 0 0.0 1 100.0 --'. Append the -- to make the end of "
      "the temperatures! A label that is not specified will be "
      "assumed to be variable and used for the computation.");
  command.SetOptionLongTag("Temperatures", "temperature-label-pairs");
  command.AddOptionField("Temperatures", "temperature", MetaCommand::LIST, true);

  command.SetOption("Iterations", "i", false, "Specify the number of iterations (default 1) the code is run.");
  command.AddOptionField("Iterations", "iterations", MetaCommand::INT, true);

  // supersample the input (2 means 4 times more voxel)
  command.SetOption("SuperSample", "s", false, "Specify the number up-sampling steps using nearest neighboor interpolation (0).");
  command.AddOptionField("SuperSample", "supersample", MetaCommand::INT, true);

  // quantize the output temperature
  command.SetOption("Quantize", "q", false, "Quantize the output into N different regions.");
  command.AddOptionField("Quantize", "quantize", MetaCommand::INT, true);

  if (!command.Parse(argc, argv)) {
    return 1;
  }

  std::string input = command.GetValueAsString("infile");
  std::string outdir = command.GetValueAsString("outdir");
  fprintf(stdout, "input: \"%s\"\n", input.c_str());
  fprintf(stdout, "outdir: \"%s\"\n", outdir.c_str());
  if (!boost::filesystem::exists(input)) {
    std::cout << "Could not find the input file..." << std::endl;
    exit(1);
  }

  int supersampling = 0;
  if (command.GetOptionWasSet("SuperSample"))
    supersampling = command.GetValueAsInt("SuperSample", "supersample");
  if (supersampling > 10) {
    fprintf(stdout, "Error: too much super-sampling...\n");
    exit(0);
  }
  if (supersampling < 0) {
    fprintf(stdout, "Error: don't know how to supersample that...\n");
    exit(-1);
  }
  int quantize = -1; // don't quantize
  if (command.GetOptionWasSet("Quantize"))
    quantize = command.GetValueAsInt("Quantize", "quantize");

  // todo: instead of number of iterations it would be good to have convergence error (might require double computations)
  int iterations = 1;
  if (command.GetOptionWasSet("Iterations"))
    iterations = command.GetValueAsInt("Iterations", "iterations");

  std::map<int, float> temperatureByMaterial;

  std::vector<int> labels;
  std::vector<float> temperatures;

  std::string temp_str = ""; // > 0
  if (command.GetOptionWasSet("Temperatures")) {
    std::list<std::string> thresholds = command.GetValueAsList("Temperatures");
    std::list<std::string>::iterator it;
    if (thresholds.size() % 2 != 0) {
      fprintf(stdout, "Error: should be an even number of threshold values and temperatures but found %lu entries.\n", thresholds.size());
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
    output_filename2 = fn + "_gradient.nrrd";
  else
    output_filename2 = fn.substr(0, lastdot) + "_gradient.nrrd";

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

  // do supersampling if required
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
  std::fill(output.begin(), output.end(), 0.0);
  data.resize(dims[0] * dims[1] * dims[2]); // the labels as int
  itk::ImageRegionIterator<ImageType> volIterator(inputVol, region);
  size_t counter = 0;
  while (!volIterator.IsAtEnd()) {
    data[counter] = volIterator.Get();
    counter++;
    ++volIterator;
  }
  tmpData.resize(dims[0] * dims[1] * dims[2]); // temporary temperatures as float
  std::fill(tmpData.begin(), tmpData.end(), 0.0);

  // run the iterations
  for (int i = 0; i < iterations; i++) {
    fprintf(stdout, "step: %d/%d\n", i + 1, iterations);
    oneStep(dims, temperatureByMaterial);
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

  // export the potential field
  typedef itk::ImageFileWriter<GradientImageType> GradientWriterType;
  GradientWriterType::Pointer writer2 = GradientWriterType::New();
  // check if that directory exists, create before writing
  writer2->SetFileName(resultJSON["output_gradient"]);
  writer2->SetInput(gimage);

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
    std::vector<double> quartiles(quantize);
    // now set the threshold temperature for each quartile range
    for (int i = 0; i < quantize; i++) {
      // what is the first temperature where we reach the current quantile?
      double quant_step = i * (1.0 / quantize); // lower border of quantile
      for (int j = 0; j < h_size; j++) {
        if (cum_hist[j] > quant_step) {
          quartiles[i] = (j / h_size) * (maxTemp - minTemp) + minTemp; // temperature at this index
          break;
        }
      }
    }
    temperatureIterator.GoToBegin();
    itk::ImageRegionIterator<OutputImageType> outputIterator(outQuant, region);
    while (!temperatureIterator.IsAtEnd() && !maskIterator.IsAtEnd() && !outputIterator.IsAtEnd()) {
      outputIterator.Set(0); // outside
      if (maskIterator.Get() != 0) {
        if (temperatureByMaterial.find(maskIterator.Get()) == temperatureByMaterial.end()) {
          // this label does not have a fixed temperature, lets use it
          // what is the quantile for this voxel?
          for (int i = 0; i < quartiles.size(); i++) {
            if (temperatureIterator.Get() > quartiles[i]) {
              outputIterator.Set(i + 1); // start counting from 1
              break;
            }
          }
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

  std::ostringstream o;
  std::string si(output_filename);
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
