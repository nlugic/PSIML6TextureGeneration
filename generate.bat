for /f %%f in ('dir /b .\trained_models\') do python generate.py .\trained_models\%%f -o .\generated_textures\ -r 2 -c 2