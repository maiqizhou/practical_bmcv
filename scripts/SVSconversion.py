import os
import pyvips

input_folder = "BCIFiltered/HE/train"
output_folder = "BCIFiltered/HE/train_1"

# Define the magnification and resolution
magnification = 20  # Assuming 20x magnification
resolution = 0.46  # Resolution in micrometers per pixel

for filename in os.listdir(input_folder):
    if filename.endswith(".svs"):
        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, filename.replace('.svs', '.png'))

        image = pyvips.Image.new_from_file(input_file, access="sequential")
        image = image.copy()
        image.set_type(pyvips.GValue.gstr_type, "aperio.AppMag", str(magnification))
        image.tiffsave(output_file, tile=True, pyramid=True, compression='jpeg', subifd=2)


print("Conversion complete.")
