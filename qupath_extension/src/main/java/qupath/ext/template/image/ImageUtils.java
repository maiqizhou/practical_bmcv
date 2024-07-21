package qupath.ext.template.image;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import qupath.lib.images.servers.ImageServer;

public class ImageUtils {
	public static String getImageDetails(ImageServer<?> server) {
        return String.format("Width: %d\nHeight: %d\nNumber of channels: %d\nPixel type: %s\n", 
                                server.getWidth(), 
                                server.getHeight(), 
                                server.nChannels(), 
                                server.getPixelType()
                                );
}
    public static boolean saveImageToFile(BufferedImage image, String filePath) {
        if(image == null || filePath == null || filePath.trim().isEmpty()) {
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
}

