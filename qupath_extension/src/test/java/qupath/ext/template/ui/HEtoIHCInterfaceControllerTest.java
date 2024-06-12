package qupath.ext.template.ui;

import org.testfx.framework.junit5.ApplicationTest;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.image.ImageView;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;

import static org.junit.jupiter.api.Assertions.assertNotNull;

import org.junit.jupiter.api.Test;

public class HEtoIHCInterfaceControllerTest extends ApplicationTest {

	@Override
	public void start(Stage stage) throws Exception {
	    FXMLLoader loader = new FXMLLoader(getClass().getResource("/qupath/ext/template/ui/interface.fxml"));
	    VBox vbox = new VBox();
	    loader.setRoot(vbox);
	    // Set controller
	    HEtoIHCInterfaceController controller = new HEtoIHCInterfaceController();
	    loader.setController(controller);
	  
	    // Load the FXML file into the VBox
	    loader.load();

	    Scene scene = new Scene(vbox);
	    stage.setScene(scene);
	    stage.show();
	}



    @Test
    public void loadImageTest() {
        // Click on the button to load the HE image
        clickOn("#loadImageButton");
        
        // Verify that the HE image is loaded
        ImageView heImageView = lookup("#heImageView").query();
        assertNotNull(heImageView.getImage(), "HE image should be loaded");

        // Verify that the IHC image is generated and displayed
        ImageView ihcImageView = lookup("#ihcImageView").query();
        assertNotNull(ihcImageView.getImage(), "IHC image should be generated and displayed");
    }
	}
