
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace WildWatch.Services
{
    public class SpeciesPrediction
    {
        public string Label { get; set; } = "";
        public float Score { get; set; }
    }

    public class SpeciesNetService : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly List<string> _labels;

        public SpeciesNetService()
        {
            string basePath = AppContext.BaseDirectory;

            string modelPath = Path.Combine(
                basePath,
                "Models",
                "speciesnet-v4.0.3b-full_image.fp16.onnx"
            );

            string labelsPath = Path.Combine(
                basePath,
                "Models",
                "labels.txt"
            );

            if (!File.Exists(modelPath))
            {
                throw new FileNotFoundException(
                    "ONNX model not found.",
                    modelPath
                );
            }

            if (!File.Exists(labelsPath))
            {
                throw new FileNotFoundException(
                    "labels.txt not found.",
                    labelsPath
                );
            }

            _labels = File.ReadAllLines(labelsPath).ToList();

            _session = new InferenceSession(modelPath);

            Console.WriteLine(
                "SpeciesNet model loaded successfully!"
            );

            Console.WriteLine(
                $"Number of labels: {_labels.Count}"
            );
        }

        public List<SpeciesPrediction> Predict(string imagePath)
        {
            if (!File.Exists(imagePath))
            {
                throw new FileNotFoundException(
                    "Image not found.",
                    imagePath
                );
            }

            using Image<Rgb24> image =
                Image.Load<Rgb24>(imagePath);

            // Step 1: Crop the central vertical portion.
            int originalWidth = image.Width;
            int originalHeight = image.Height;

            int cropHeight = Math.Max(
                (int)(originalHeight * 0.7),
                originalHeight - 400
            );

            cropHeight = Math.Min(
                cropHeight,
                originalHeight
            );

            int cropY = (originalHeight - cropHeight) / 2;

            image.Mutate(ctx => ctx.Crop(
                new Rectangle(
                    0,
                    cropY,
                    originalWidth,
                    cropHeight
                )
            ));

            // Step 2: Resize to 480 x 480 using bilinear sampling.
            image.Mutate(ctx => ctx.Resize(new ResizeOptions
            {
                Size = new Size(480, 480),
                Mode = ResizeMode.Stretch,
                Sampler = KnownResamplers.Triangle
            }));

            // Step 3: Convert the image to an NHWC tensor.
            // Shape: [batch, height, width, channels]
            // Pixel values: RGB in the range [0, 1].

            var tensor = new DenseTensor<float>(
                new[] { 1, 480, 480, 3 }
            );

            for (int y = 0; y < 480; y++)
            {
                for (int x = 0; x < 480; x++)
                {
                    Rgb24 pixel = image[x, y];

                    tensor[0, y, x, 0] = pixel.R / 255f;
                    tensor[0, y, x, 1] = pixel.G / 255f;
                    tensor[0, y, x, 2] = pixel.B / 255f;
                }
            }

            // Step 4: Prepare ONNX input.
            string inputName =
                _session.InputMetadata.Keys.First();

            var inputs = new[]
            {
                NamedOnnxValue.CreateFromTensor(
                    inputName,
                    tensor
                )
            };

            // Step 5: Run inference.
            using IDisposableReadOnlyCollection<
                DisposableNamedOnnxValue> results =
                    _session.Run(inputs);

            float[] logits = results
                .First()
                .AsEnumerable<float>()
                .ToArray();

            if (logits.Length != _labels.Count)
            {
                throw new InvalidOperationException(
                    $"Model returned {logits.Length} scores, " +
                    $"but labels.txt contains {_labels.Count} labels."
                );
            }

            // Step 6: Convert logits to probabilities using softmax.
            // Subtract the maximum logit for numerical stability.

            float maxLogit = logits.Max();

            double[] exponentials = logits
                .Select(value => Math.Exp(value - maxLogit))
                .ToArray();

            double total = exponentials.Sum();

            if (total <= 0 || double.IsInfinity(total)
                || double.IsNaN(total))
            {
                throw new InvalidOperationException(
                    "Unable to calculate prediction probabilities."
                );
            }

            // Step 7: Return the five highest-scoring labels.
            // Raw labels may contain UUID and taxonomy information.
            // Keep only the final semicolon-separated label for display.

            return Enumerable.Range(0, logits.Length)
                .Select(i =>
                {
                    string rawLabel = _labels[i];

                    string[] parts = rawLabel.Split(';');

                    string cleanLabel =
                        parts[parts.Length - 1].Trim();

                    return new SpeciesPrediction
                    {
                        Label = cleanLabel,
                        Score = (float)(
                            exponentials[i] / total
                        )
                    };
                })
                .OrderByDescending(p => p.Score)
                .Take(5)
                .ToList();
        }

        public void Dispose()
        {
            _session.Dispose();
        }
    }
}
