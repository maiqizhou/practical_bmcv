package qupath.ext.template.ui;

import javafx.application.Platform;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.geometry.Point2D;
import javafx.scene.control.Alert;
import javafx.scene.control.Alert.AlertType;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.TextArea;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.MouseEvent;
import javafx.scene.input.ScrollEvent;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.images.servers.ImageServerProvider;
import qupath.lib.regions.RegionRequest;
import qupath.ext.template.image.ImageUtils;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Base64;
import java.util.ResourceBundle;
import java.io.InputStream;
import java.io.OutputStream;


/**
 * Controller for UI pane contained in interface.fxml
 */
public class HEtoIHCInterfaceController extends VBox {
    private static final ResourceBundle resources = ResourceBundle.getBundle("qupath.ext.template.ui.strings");

    @FXML
    private ImageView heImageView;

    @FXML
    private ImageView ihcImageView;

    @FXML
    private TextArea heImageDetailsTextArea;
    
    @FXML
    private Label conversionStatusLabel;

    @FXML
    private TextArea ihcImageDetailsTextArea;
    
    @FXML
    private Label statusLabel;

    private ImageServer<BufferedImage> heImageServer;
    private ImageServer<BufferedImage> ihcImageServer;
    
    @FXML
    private ScrollPane scrollPane;

    private File selectedHeImageFile;

    /**
     * Create an instance of InterfaceController
     *
     * @return a new instance of InterfaceController
     * @throws IOException if the FXML file cannot be loaded
     */
    public static HEtoIHCInterfaceController createInstance() throws IOException {
        return new HEtoIHCInterfaceController();
    }

    /**
     * Private constructor to initialize the controller and load the FXML file
     *
     * @throws IOException if the FXML file cannot be loaded
     */
    public HEtoIHCInterfaceController() throws IOException {
        var url = HEtoIHCInterfaceController.class.getResource("interface.fxml");
        FXMLLoader loader = new FXMLLoader(url, resources);
        loader.setRoot(this);
        loader.setController(this);
        loader.load();
    }

    /**
     * Method to load the HE image
     */
    @FXML
    private void handleLoadImage(ActionEvent event) {
        FileChooser fileChooser = new FileChooser();
        fileChooser.getExtensionFilters().addAll(
                new FileChooser.ExtensionFilter("Image Files", "*.png", "*.jpg")
        );
        selectedHeImageFile = fileChooser.showOpenDialog(null);

        if (selectedHeImageFile != null) {
            
            ImageUtils.checkImageFormat(selectedHeImageFile.getAbsolutePath());
            if (selectedHeImageFile == null) {
                return; 
            }

            try (FileInputStream fileInputStream = new FileInputStream(selectedHeImageFile)) {
                Image heImage = new Image(fileInputStream);
                heImageView.setImage(heImage);
                addDragAndZoomFunctionality(heImageView);
                ihcImageView.setImage(null); // Clear the IHC image view
                conversionStatusLabel.setText("");
            } catch (IOException e) {
                showAlert("Error", "Failed to load image: " + e.getMessage());
            }

            try {
                heImageServer = ImageServerProvider.buildServer(selectedHeImageFile.getAbsolutePath(), BufferedImage.class);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * Method to convert image
     */
    @FXML
    private void handleConvertImage(ActionEvent event) {
        if (selectedHeImageFile != null) {
            convertToIHC(selectedHeImageFile);
        } else {
            showAlert("Error", "No HE image loaded to convert.");
        }
    }

    public void convertToIHC(File heImageFile) {
        // Set the conversionStatusLabel to "Conversion in progress" immediately
        Platform.runLater(() -> conversionStatusLabel.setText("Conversion in progress"));

        new Thread(() -> {
            try {
                // Path to the Python script
                String pythonScriptPath = "/Users/maggie/practical_bmcv/transform_test.py";

                // Command to run the Python script
                ProcessBuilder processBuilder = new ProcessBuilder(
                    "/Users/maggie/anaconda3/bin/python",
                    pythonScriptPath,
                    heImageFile.getAbsolutePath()
                );
                processBuilder.redirectErrorStream(true);

                Process process = processBuilder.start();
                BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
                String line;
                String lastLine = null;
                while ((line = reader.readLine()) != null) {
                    final String status = line;
                    // Regular expression to check if a string might be Base64
                    if (!status.matches("^[a-zA-Z0-9/+]*={0,2}$")) {
                        Platform.runLater(() -> statusLabel.setText(""));
                    }
                    lastLine = line;
                }
                process.waitFor();

                // The last line should be the Base64 string
                String ihcImageBase64 = lastLine;
                byte[] imageBytes = Base64.getDecoder().decode(ihcImageBase64);

                // Save the decoded image to a temporary file
                File tempFile = File.createTempFile("ihcImage", ".png");
                try (OutputStream out = new FileOutputStream(tempFile)) {
                    out.write(imageBytes);
                }

                // Create a new ImageServer that loads the temporary file
                ihcImageServer = ImageServerProvider.buildServer(tempFile.toURI().toString(), BufferedImage.class);

                try (InputStream in = new ByteArrayInputStream(imageBytes)) {
                    Image ihcImage = new Image(in);
                    Platform.runLater(() -> {
                        ihcImageView.setImage(ihcImage);
                        // Add zoom functionality
                        addDragAndZoomFunctionality(ihcImageView);
                        // Set the conversionStatusLabel to "Conversion completed"
                        conversionStatusLabel.setText("Conversion completed");
                    });
                }
            } catch (IOException | InterruptedException e) {
                Platform.runLater(() -> showAlert("Error", "Failed to convert image: " + e.getMessage()));
            }
        }).start();
    }






    /**
     * Method to show details of the image
     */
    @FXML
    private void showHeImageDetails(ActionEvent event) {
        if (heImageView.getImage() != null) {
            String imageDetails = "HE Image:\n" + ImageUtils.getImageDetails(heImageServer);
            heImageDetailsTextArea.setText(imageDetails);
        } else {
            heImageDetailsTextArea.setText("");
        }
    }

    @FXML
    private void showIhcImageDetails(ActionEvent event) {
        if (ihcImageView.getImage() != null) {
            String imageDetails = "IHC Image:\n" + ImageUtils.getImageDetails(ihcImageServer);
            ihcImageDetailsTextArea.setText(imageDetails);
        } else {
            ihcImageDetailsTextArea.setText("No IHC image to show details.");
        }
    }

    @SuppressWarnings("deprecation")
    @FXML
    private void handleDownloadHeImage(ActionEvent event) {
        if (heImageServer == null) {
            showAlert("Error", "No HE image loaded to save.");
            return;
        }

        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Save Image");
        fileChooser.getExtensionFilters().add(
                new FileChooser.ExtensionFilter("PNG", "*.png"));
        File file = fileChooser.showSaveDialog(null);

        if (file != null) {
            try {
                BufferedImage image = heImageServer.readBufferedImage(RegionRequest.createInstance(heImageServer.getPath(), 1, 0, 0, heImageServer.getWidth(), heImageServer.getHeight()));
                ImageUtils.saveImageToFile(image, file.getAbsolutePath());
            } catch (IOException e) {
                showAlert("Error", "Failed to save image: " + e.getMessage());
            }
        }
    }

    @SuppressWarnings("deprecation")
    @FXML
    private void handleDownloadIhcImage(ActionEvent event) {
        if (ihcImageServer == null) {
            showAlert("Error", "No IHC image loaded to save.");
            return;
        }

        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Save Image");
        fileChooser.getExtensionFilters().add(
                new FileChooser.ExtensionFilter("PNG", "*.png"));
        File file = fileChooser.showSaveDialog(null);

        if (file != null) {
            try {
                BufferedImage image = ihcImageServer.readBufferedImage(RegionRequest.createInstance(ihcImageServer.getPath(), 1, 0, 0, ihcImageServer.getWidth(), ihcImageServer.getHeight()));
                ImageUtils.saveImageToFile(image, file.getAbsolutePath());
            } catch (IOException e) {
                showAlert("Error", "Failed to save image: " + e.getMessage());
            }
        }
    }

    public void addDragAndZoomFunctionality(ImageView imageView) {
        // Add drag functionality
        imageView.setOnMousePressed(new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent event) {
                // Record the current mouse position
                imageView.setUserData(new Point2D(event.getX(), event.getY()));
            }
        });

        imageView.setOnMouseDragged(new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent event) {
                // Calculate the mouse movement
                Point2D oldMousePosition = (Point2D) imageView.getUserData();
                double deltaX = event.getX() - oldMousePosition.getX();
                double deltaY = event.getY() - oldMousePosition.getY();

                // Update the ImageView position
                imageView.setTranslateX(imageView.getTranslateX() + deltaX);
                imageView.setTranslateY(imageView.getTranslateY() + deltaY);

                // Update the recorded mouse position
                imageView.setUserData(new Point2D(event.getX(), event.getY()));
            }
        });

        // Add zoom functionality
        imageView.setOnScroll(new EventHandler<ScrollEvent>() {
            @Override
            public void handle(ScrollEvent event) {
                double zoomFactor = 1.05;
                double deltaY = event.getDeltaY();
                if (deltaY < 0) {
                    zoomFactor = 1 / zoomFactor;
                }
                imageView.setScaleX(imageView.getScaleX() * zoomFactor);
                imageView.setScaleY(imageView.getScaleY() * zoomFactor);
                event.consume();
            }
        });
    }
    public void showAlert(String title, String message) {
        Alert alert = new Alert(AlertType.ERROR);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }
}
