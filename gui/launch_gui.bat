if exist CodeLabellingGUI.jar (
    cd ..
    start java -jar gui/CodeLabellingGUI.jar
    start python cesres_graphcodebert_model.py -c gui/config_gui.yaml
    cd gui
) else (
    ECHO "Please run this script when inside the directory containing it."
)