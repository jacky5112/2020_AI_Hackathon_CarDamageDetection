using System;
using System.Collections.Generic;
using System.Web.Http;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Drawing.Imaging;
using System.Web.Configuration;

namespace _2_WebApiServer.App_Code
{
    public class CarDamageDetectionController : ApiController
    {
        // GET api/<controller>
        public IEnumerable<string> Get()
        {
            return new string[] { "ImageData" };
        }

        // GET api/<controller>/5
        public string Get(string image_data)
        {
            return null;
        }

        // POST api/<controller>
        [HttpPost]
        public OutputObjectDetection Post([FromBody] InputObjectDetection value)
        {
            try
            {
                OutputObjectDetection ret = new OutputObjectDetection();
                string ticks = DateTime.Now.Ticks.ToString();
                string webRoot = System.Web.Hosting.HostingEnvironment.MapPath("~/");
                string execRoot = string.Format(@"{0}Exec", webRoot);
                string execFullPath = string.Format(@"{0}\1_Maskrcnn_Segmentation.exe", execRoot);
                string imageSrcName = @"ImageSrc";
                string imageSrcRoot = string.Format(@"{0}\{1}", execRoot, imageSrcName);
                string imageSrcPath = string.Format(@"{0}\{1}.jpg", imageSrcName, ticks);
                string imageSrcFullPath = string.Format(@"{0}\{1}.jpg", imageSrcRoot, ticks);
                string imageDstName = @"ImageDst";
                string imageDstRoot = string.Format(@"{0}\{1}", execRoot, imageDstName);
                string imageDstPath = string.Format(@"{0}\{1}.jpg", imageDstName, ticks);
                string imageDstFullPath = string.Format(@"{0}\{1}.jpg", imageDstRoot, ticks);

                // init
                ret.ObjectDetail = new List<ObjectDetectionDetail>();

                // Try to Write Image
                byte[] image_bytes = Convert.FromBase64String(value.ImageData);
                Image image;

                using (MemoryStream ms = new MemoryStream(image_bytes))
                {
                    image = Image.FromStream(ms);

                    using (MemoryStream image_ms = new MemoryStream())
                    {
                        image.Save(image_ms, ImageFormat.Jpeg);

                        using (FileStream fs = File.Open(imageSrcFullPath, FileMode.OpenOrCreate))
                        {
                            fs.Write(ms.ToArray(), 0, ms.ToArray().Length);
                            fs.Flush();
                        } // end using
                    } // end using
                } // end using

                // GDI+ Error  !!!
                //image.Save(imageSrcFullPath, ImageFormat.Jpeg);

                // Execute object detecion
                // Note. --image={圖片目錄} --weight=model\frozen_inference_graph.pb --graph=model\maskrcnn_inception_20201008.pbtxt --classes=car_labels.names --colors=colors.txt --output_file={輸出名稱} --outlayer_names=detection_out_final,detection_masks --backend=OpenCV --target=CPU --mask=0.3 --conf=0.7
                Process process = new Process();

                process.StartInfo.FileName = execFullPath;
                process.StartInfo.Arguments = string.Format(@"--image={0} --weight=model\frozen_inference_graph.pb --graph=model\maskrcnn_inception_20201008.pbtxt --classes=car_labels.names --colors=colors.txt --output_file={1} --outlayer_names=detection_out_final,detection_masks --backend=OpenCV --target=CPU --mask=0.3 --conf=0.7", imageSrcPath, imageDstPath);
                process.StartInfo.WorkingDirectory = execRoot;
                process.StartInfo.UseShellExecute = false;
                process.StartInfo.RedirectStandardOutput = true;
                process.StartInfo.CreateNoWindow = true;

                process.Start();
                process.WaitForExit(30 * 1000);

                string result = process.StandardOutput.ReadToEnd().Replace("\r", "");
                string[] split_result = result.Split(new string[] { "\n" }, StringSplitOptions.RemoveEmptyEntries);

                foreach (string i in split_result)
                {
                    // [*] {"Frame":0,"X":49,"Y":52,"W":162,"H":131,"Label":glass,Score:1.00}
                    // Frame -> if you use "image", always "0"
                    if (i.Contains("[*]"))
                    {
                        ObjectDetectionDetail detection = new ObjectDetectionDetail();
                        string tmp = i.Replace("[*] ", "").Replace("{", "").Replace("}", "");
                        string[] element = tmp.Split(new string[] { "," }, StringSplitOptions.RemoveEmptyEntries);
                        
                        foreach (string j in element)
                        {
                            
                            string[] data = j.Split(new string[] { ":" }, StringSplitOptions.RemoveEmptyEntries);

                            if (data[0].Contains("Frame"))
                            {
                                // image always '0'
                                continue;
                            } // end if
                            else if (data[0].Contains("X"))
                            {
                                detection.X = int.Parse(data[1]);
                            } // end else if
                            else if (data[0].Contains("Y"))
                            {
                                detection.Y = int.Parse(data[1]);
                            } // end else if
                            else if (data[0].Contains("W"))
                            {
                                detection.W = int.Parse(data[1]);
                            } // end else if
                            else if (data[0].Contains("H"))
                            {
                                detection.H = int.Parse(data[1]);
                            } // end else if
                            else if (data[0].Contains("Label"))
                            {
                                detection.Label = data[1];
                            } // end else if
                            else if (data[0].Contains("Score"))
                            {
                                detection.Score = double.Parse(data[1]);
                            } // end else if
                        } // end for

                        ret.ObjectDetail.Add(detection);
                    } // end if
                } // end for

                // Try to Read Image
                using (MemoryStream ms = new MemoryStream())
                {
                    image = Image.FromFile(imageDstFullPath);
                    image.Save(ms, ImageFormat.Jpeg);
                    image_bytes = ms.ToArray();
                    ret.ResultImageData = Convert.ToBase64String(image_bytes);
                    image.Dispose();
                } // end using

                return ret;
            } // end try
            catch (Exception ex)
            {
                return null;
            } // end catch
        }

        // PUT api/<controller>/5
        public void Put()
        {
        }

        // DELETE api/<controller>/5
        public void Delete()
        {
        }

        public class InputObjectDetection
        {
            public string ImageData { get; set; }
        }

        // Example: [*] {"Frame":0,"X":49,"Y":52,"W":162,"H":131,"Label",glass,Score:1.00}
        public class ObjectDetectionDetail
        {
            public int X { get; set; }
            public int Y { get; set; }
            public int W { get; set; }
            public int H { get; set; }
            public string Label { get; set; }
            public double Score { get; set; }
        }

        public class OutputObjectDetection
        {
            public List<ObjectDetectionDetail> ObjectDetail { get; set; }
            public string ResultImageData;
        } // end class
    }
}