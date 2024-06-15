package qupath.ext.template.image;

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


}
