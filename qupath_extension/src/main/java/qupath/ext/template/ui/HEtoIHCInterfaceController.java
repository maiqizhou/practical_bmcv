package qupath.ext.template.ui;

import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.control.Alert;
import javafx.scene.control.Alert.AlertType;
import javafx.scene.control.Spinner;
import javafx.scene.control.SpinnerValueFactory;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;
import qupath.ext.template.HEtoIHCExtension;
import qupath.fx.dialogs.Dialogs;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ResourceBundle;

/**
 * Controller for UI pane contained in interface.fxml
 */
public class HEtoIHCInterfaceController extends VBox {
	private static final ResourceBundle resources = ResourceBundle.getBundle("qupath.ext.template.ui.strings");

	@FXML
	public Spinner<Integer> threadSpinner;

	@FXML
	public void initialize() {
		SpinnerValueFactory<Integer> valueFactory = new SpinnerValueFactory.IntegerSpinnerValueFactory(1, 100, 1);
		threadSpinner.setValueFactory(valueFactory);
	}

	@FXML
	private ImageView heImageView;

	@FXML
	private ImageView ihcImageView;
	
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

		// Bind the spinner value to the number of threads property in DemoExtension
		threadSpinner.getValueFactory().valueProperty().bindBidirectional(HEtoIHCExtension.numThreadsProperty());
		threadSpinner.getValueFactory().valueProperty().addListener((observableValue, oldValue, newValue) -> {
			Dialogs.showInfoNotification(resources.getString("title"),
					String.format(resources.getString("threads"), newValue));
		});
	}


	/**
	 * Method to load the HE image
	 */
    @FXML
    private void handleLoadImage(ActionEvent event) {
        FileChooser fileChooser = new FileChooser();
        fileChooser.getExtensionFilters().addAll(
            new FileChooser.ExtensionFilter("Image Files", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
        );
        selectedHeImageFile = fileChooser.showOpenDialog(null);

        if (selectedHeImageFile != null) {
            try (FileInputStream fileInputStream = new FileInputStream(selectedHeImageFile)) {
                Image heImage = new Image(fileInputStream);
                heImageView.setImage(heImage);
                ihcImageView.setImage(null); // Clear the IHC image view
            } catch (IOException e) {
                showAlert("Error", "Failed to load image: " + e.getMessage());
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
        try {
            // Path to the Python script
            String pythonScriptPath = "/Users/maggie/Downloads/BCI-main/transform_test.py";

            // Command to run the Python script
            ProcessBuilder processBuilder = new ProcessBuilder("/Users/maggie/anaconda3/bin/python", pythonScriptPath, heImageFile.getAbsolutePath());
            processBuilder.redirectErrorStream(true);

            Process process = processBuilder.start();
            process.waitFor();

            // Assume the Python script saves the converted IHC image to a predefined location
            File ihcImageFile = new File("/Users/maggie/Downloads/BCI-main/ihc_image.png");

            // Load the converted IHC image
            if (ihcImageFile.exists()) {
                try (FileInputStream fileInputStream = new FileInputStream(ihcImageFile)) {
                    Image ihcImage = new Image(fileInputStream);
                    ihcImageView.setImage(ihcImage);
                }
            } else {
                showAlert("Error", "Converted IHC image not found");
            }
        } catch (IOException | InterruptedException e) {
            showAlert("Error", "Failed to convert image: " + e.getMessage());
        }
    }

    public void showAlert(String title, String message) {
        Alert alert = new Alert(AlertType.ERROR);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }
}

