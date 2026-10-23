
namespace WildWatch
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            // Register services
            builder.Services.AddControllers();
            builder.Services.AddEndpointsApiExplorer();
            builder.Services.AddSwaggerGen();

            builder.Services.AddSingleton<
                WildWatch.Services.SpeciesNetService>();

            var app = builder.Build();

            // Load SpeciesNet model at startup
            using (var scope = app.Services.CreateScope())
            {
                var speciesNet = scope.ServiceProvider
                    .GetRequiredService<
                        WildWatch.Services.SpeciesNetService>();
            }

            // Enable Swagger UI in Development
            if (app.Environment.IsDevelopment())
            {
                app.UseSwagger();
                app.UseSwaggerUI();
            }

            app.UseHttpsRedirection();
            app.UseAuthorization();
            app.MapControllers();

            app.Run();
        }
    }
}
