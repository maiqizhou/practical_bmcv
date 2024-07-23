package qupath.ext.template.image;

import javafx.application.Platform;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.ButtonType;
import javafx.scene.control.DialogPane;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;
import javafx.stage.Window;
import org.junit.jupiter.api.Test;
import org.testfx.framework.junit5.ApplicationTest;
import org.testfx.util.WaitForAsyncUtils;
import qupath.ext.template.ui.HEtoIHCInterfaceController;

import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ImageUtilsTest extends ApplicationTest {

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
    public void testValidImage() {
        String validImagePath = "/Users/maggie/Downloads/BCI-main/PyramidPix2pix/BCI_dataset/HE/train/02528_train_3+.png";
        runAndWait(() -> ImageUtils.checkImageFormat(validImagePath));
        // Check that no alert is shown for a valid image
    }

    @Test
    public void testInvalidImageFormat() {
        String invalidImagePath = "/Users/maggie/practical_bmcv/resultsHE.tsv"; // A text file pretending to be an image
        runAndWait(() -> ImageUtils.checkImageFormat(invalidImagePath));
        Optional<ButtonType> result = waitForAlert();
        assertTrue(result.isPresent());
        assertEquals(ButtonType.OK, result.get());
    }

    @Test
    public void testFileNotFound() {
        String nonExistentImagePath = "/Users/maggie/practical_bmcv/resultHE.tsv";
        runAndWait(() -> ImageUtils.checkImageFormat(nonExistentImagePath));
        Optional<ButtonType> result = waitForAlert();
        assertTrue(result.isPresent());
        assertEquals(ButtonType.OK, result.get());
    }

    private Optional<ButtonType> waitForAlert() {
        WaitForAsyncUtils.waitForFxEvents();
        return getActiveAlert().map(Alert::getResult);
    }

    private Optional<Alert> getActiveAlert() {
        return Window.getWindows().stream()
                .filter(window -> window instanceof Stage)
                .map(window -> (Stage) window)
                .filter(stage -> stage.isShowing())
                .map(stage -> stage.getScene().getRoot())
                .filter(root -> root instanceof DialogPane)
                .map(root -> (DialogPane) root)
                .map(DialogPane::getScene)
                .map(Scene::getWindow)
                .filter(window -> window instanceof Stage)
                .map(window -> (Stage) window)
                .map(stage -> stage.getProperties().get("alert"))
                .filter(alert -> alert instanceof Alert)
                .map(alert -> (Alert) alert)
                .findFirst();
    }

    private void runAndWait(Runnable action) {
        try {
            CountDownLatch latch = new CountDownLatch(1);
            Platform.runLater(() -> {
                try {
                    action.run();
                } finally {
                    latch.countDown();
                }
            });
            latch.await(10, TimeUnit.SECONDS); // Adjust timeout as needed
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }
}
