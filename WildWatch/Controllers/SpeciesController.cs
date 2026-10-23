
using Microsoft.AspNetCore.Mvc;
using WildWatch.Services;

namespace WildWatch.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class SpeciesController : ControllerBase
    {
        private readonly SpeciesNetService _speciesNet;

        public SpeciesController(SpeciesNetService speciesNet)
        {
            _speciesNet = speciesNet;
        }

        [HttpPost("predict")]
        public IActionResult Predict(IFormFile image)
        {
            if (image == null || image.Length == 0)
            {
                return BadRequest(new
                {
                    message = "Please upload an image."
                });
            }

            string extension = Path.GetExtension(image.FileName);

            string[] allowedExtensions =
            {
                ".jpg", ".jpeg", ".png", ".webp"
            };

            if (!allowedExtensions.Contains(
                extension,
                StringComparer.OrdinalIgnoreCase))
            {
                return BadRequest(new
                {
                    message = "Upload a JPG, PNG, or WebP image."
                });
            }

            string tempPath = Path.Combine(
                Path.GetTempPath(),
                Guid.NewGuid().ToString() + extension);

            try
            {
                using (var stream = System.IO.File.Create(tempPath))
                {
                    image.CopyTo(stream);
                }

                var predictions = _speciesNet.Predict(tempPath);

                return Ok(new
                {
                    fileName = Path.GetFileName(image.FileName),
                    predictions = predictions.Select(p => new
                    {
                        label = p.Label,
                        confidence = Math.Round(p.Score * 100, 2)
                    })
                });
            }
            finally
            {
                if (System.IO.File.Exists(tempPath))
                {
                    System.IO.File.Delete(tempPath);
                }
            }
        }
    }
}
