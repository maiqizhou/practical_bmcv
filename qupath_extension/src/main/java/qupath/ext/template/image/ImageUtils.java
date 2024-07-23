package qupath.ext.template.image;

import javafx.scene.control.Alert;
import javafx.scene.control.Alert.AlertType;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

import qupath.lib.images.servers.ImageServer;
import qupath.lib.images.servers.ImageServerMetadata;
import qupath.lib.images.servers.PixelCalibration;

public class ImageUtils {


	    public static String getImageDetails(ImageServer<?> server) {
	        ImageServerMetadata metadata = server.getMetadata();
	        PixelCalibration pixelCalibration = server.getPixelCalibration();
	        StringBuilder details = new StringBuilder();
	        details.append(String.format("Width: %d pixels\n", server.getWidth()));
	        details.append(String.format("Height: %d pixels\n", server.getHeight()));
	        details.append(String.format("Number of Channels: %d\n", server.nChannels()));
	        details.append(String.format("Server Type: %s\n", server.getServerType()));
	        details.append(String.format("Number of Resolutions: %d\n", server.nResolutions()));
	        details.append(String.format("Number of Timepoints: %d\n", server.nTimepoints()));
	        details.append(String.format("Number of Z-Slices: %d\n", server.nZSlices()));
	        details.append(String.format("Is RGB: %b\n", server.isRGB()));

	        if (pixelCalibration != null) {
	            details.append(String.format("Pixel Width: %.2f microns\n", pixelCalibration.getPixelWidth()));
	            details.append(String.format("Pixel Height: %.2f microns\n", pixelCalibration.getPixelHeight()));
	        }

	        if (metadata != null) {
	            details.append(String.format("Metadata - Name: %s\n", metadata.getName()));
	        }

	        details.append(String.format("Path: %s\n", server.getPath()));
	        details.append(String.format("Associated Images: %s\n", String.join(", ", server.getAssociatedImageList())));
	        details.append(String.format("Preferred Downsamples: %s\n", arrayToString(server.getPreferredDownsamples())));

	        return details.toString();
	    }

	    private static String arrayToString(double[] array) {
	        StringBuilder sb = new StringBuilder();
	        for (double d : array) {
	            sb.append(String.format("%.2f", d)).append(" ");
	        }
	        return sb.toString().trim();
	    }

    public static boolean saveImageToFile(BufferedImage image, String filePath) {
        if (image == null || filePath == null || filePath.trim().isEmpty()) {
            return false;
        }

        File outputfile = new File(filePath);

        try {
            ImageIO.write(image, "png", outputfile);
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }

        return true;
    }

    public static void checkImageFormat(String imagePath) {
        File file = new File(imagePath);
        if (!file.exists()) {
            showFileNotFoundAlert("Image file not found.");
            return;
        }

        try (FileInputStream input = new FileInputStream(file)) {
            BufferedImage image = ImageIO.read(input);

            if (image == null) {
                showFormatNotSupportedAlert("Image format not supported.");
            } else {
                // Image loaded successfully, proceed with your logic
                System.out.println("Image loaded successfully: " + imagePath);
            }
        } catch (IOException e) {
            showFormatNotSupportedAlert("An error occurred while loading the image: " + e.getMessage());
        }
    }

    private static void showFileNotFoundAlert(String message) {
        Alert alert = new Alert(AlertType.ERROR);
        alert.setTitle("File Not Found");
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }

    private static void showFormatNotSupportedAlert(String message) {
        Alert alert = new Alert(AlertType.ERROR);
        alert.setTitle("Format Not Supported");
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }
}
