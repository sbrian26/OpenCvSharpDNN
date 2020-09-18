using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OpenCVCSharpDNN.Impl
{ 

    /// <summary>
    /// Implementation of the YoloV3
    /// </summary>
    public class NetYoloV3 : NetCustom
    {
        /// <summary>
        /// Scale for the blob
        /// </summary>
        public double Scale { get; set; }

        /// <summary>
        /// The prefix of the out layer result
        /// </summary>
        const int Prefix = 5;

        /// <summary>
        /// Values of the config
        /// </summary>
        private string[] valuesConfig;


        protected override NetResult[] BeginDetect(Bitmap img, float minProbability = 0.3F, string[] labelsFilters = null, float nmsThreshold = 0.3F)
        {

            //Extract width and height from config file
            ExtractValueFromConfig("width", out int widthBlob);
            ExtractValueFromConfig("height", out int heightBlob);

            using (Mat mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(img))
            {

                //Create the blob
                using (var blob = CvDnn.BlobFromImage(mat, Scale, size: new OpenCvSharp.Size(widthBlob, heightBlob), crop: false))
                {
                    //Set blob of a default layer
                    network.SetInput(blob);

                    //Get all out layers
                    string[] outLayers = network.GetUnconnectedOutLayersNames();

                    //Initialize all blobs for the all out layers
                    // The long way:
                    //var result = new Mat[outLayers.Length];
                    //for (int i = 0; i < result.Length; i++) {
                    //    result[i] = new Mat();
                    //}
                    // The short way
                    var result = outLayers.Select(_ => new Mat()).ToArray();
                    
                    ///Execute all out layers
                    network.Forward(result, outLayers);

                    var yolos = GetResult(result, mat, minProbability, nmsThreshold);


                    //Build NetResult
                    List<NetResult> netResults = new List<NetResult>();
                    foreach (var i in yolos.indices)
                    {
                        var box = yolos.boxes[i];
                        //Draw(image, classIds[i], confidences[i], probabilities[i], box.X, box.Y, box.Width, box.Height);
                        netResults.Add(NetResult.Build((int) box.X, (int) box.Y, (int) box.Width, (int) box.Height, Labels[yolos.classIds[i]], yolos.probabilities[i] ));
                    }
                    
                    #region deprecated
                    //Build NetResult
                    //foreach (var item in result)
                    //{
                    //    for (int i = 0; i < item.Rows; i++)
                    //    {
                    //        //Get the max loc and max of the col range by prefix result
                    //        Cv2.MinMaxLoc(item.Row(i).ColRange(Prefix, item.Cols), out double min, out double max, out OpenCvSharp.Point minLoc, out OpenCvSharp.Point maxLoc);

                    //        //Validate the min probability
                    //        if (max >= minProbability)
                    //        {
                    //            //The label is the max Loc
                    //            string label = Labels[maxLoc.X];
                    //            if (labelsFilters != null)
                    //            {
                    //                if (!labelsFilters.Contains(label))
                    //                {
                    //                    continue;
                    //                }
                    //            }




                    //            //The probability is the max value
                    //            double probability = max;

                    //            //Center BoundingBox X is the 0 index result
                    //            int centerX = Convert.ToInt32(item.At<float>(i, 0) * (float)mat.Width);
                    //            //Center BoundingBox X is the 1 index result
                    //            int centerY = Convert.ToInt32(item.At<float>(i, 1) * (float)mat.Height);
                    //            //Width BoundingBox is the 2 index result
                    //            int width = Convert.ToInt32(item.At<float>(i, 2) * (float)mat.Width);
                    //            //Height BoundingBox is the 2 index result
                    //            int height = Convert.ToInt32(item.At<float>(i, 3) * (float)mat.Height);

                    //            //Build NetResult
                    //            netResults.Add(NetResult.Build(centerX, centerY, width, height, label, probability));

                    //        }
                    //    }
                    //}
                    #endregion
                    return netResults.ToArray();
                }
            }
        }
        private static (List<Rect2d> boxes, int[] indices, List<int> classIds, List<float> confidences, List<float> probabilities) GetResult(IEnumerable<Mat> output, Mat image, float threshold, float nmsThreshold)
        {
            //for nms
            var classIds = new List<int>();
            var confidences = new List<float>();
            var probabilities = new List<float>();
            var boxes = new List<Rect2d>();

            var w = image.Width;
            var h = image.Height;
            /*
             YOLO3 COCO trainval output
             0 1 : center                    2 3 : w/h
             4 : confidence                  5 ~ 84 : class probability 
            */
            const int prefix = 5;   //skip 0~4

            foreach (var prob in output)
            {
                for (var i = 0; i < prob.Rows; i++)
                {
                    var confidence = prob.At<float>(i, 4);
                    if (confidence >= threshold)
                    {
                        //get classes probability
                        Cv2.MinMaxLoc(prob.Row(i).ColRange(prefix, prob.Cols), out _, out OpenCvSharp.Point max);
                        var classes = max.X;
                        var probability = prob.At<float>(i, classes + prefix);

                        if (probability > threshold) //more accuracy, you can cancel it
                        {
                            //get center and width/height
                            var centerX = prob.At<float>(i, 0) * w;
                            var centerY = prob.At<float>(i, 1) * h;
                            var width = prob.At<float>(i, 2) * w;
                            var height = prob.At<float>(i, 3) * h;

                            //put data to list for NMSBoxes
                            classIds.Add(classes);
                            confidences.Add(confidence);
                            probabilities.Add(probability);
                            boxes.Add(new Rect2d(centerX, centerY, width, height));
                        }
                    }
                }
            }                     

            //using non-maximum suppression to reduce overlapping low confidence box
            CvDnn.NMSBoxes(boxes, confidences, threshold, nmsThreshold, out int[] indices);

            Console.WriteLine($"NMSBoxes drop {confidences.Count - indices.Length} overlapping result.");

            //foreach (var i in indices)
            //{
            //    var box = boxes[i];
            //    Draw(image, classIds[i], confidences[i], probabilities[i], box.X, box.Y, box.Width, box.Height);
            //}
            return (boxes, indices, classIds, confidences, probabilities);
        }
        /// <summary>
        /// Initialize the model
        /// </summary>
        /// <param name="pathModel"></param>
        /// <param name="pathConfig"></param>
        protected override void InitializeModel(string pathModel, string pathConfig)
        {
            valuesConfig = File.ReadAllLines(pathConfig);

            ///Initialize darknet network
            network = CvDnn.ReadNetFromDarknet(pathConfig, pathModel);

            //Set the scale 1 / 255
            this.Scale = 0.00392;
        }

        private void ExtractValueFromConfig<t>(string item, out t value)
        {
            if (valuesConfig == null || valuesConfig.Length == 0)
                throw new NullReferenceException("The file of config has empty");

            string line = valuesConfig.FirstOrDefault(p => p.ToLower().Contains(item.ToLower()));
            if (string.IsNullOrWhiteSpace(line))
                throw new NullReferenceException($" The item {item} not exits in the file.");

            string strValue = line.Split('=')[1];

            value = (t)Convert.ChangeType(strValue, typeof(t));
        }
    }
}
