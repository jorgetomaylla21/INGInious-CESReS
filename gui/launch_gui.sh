if [ -f "CodeLabellingGUI.jar" ]; then
    cd ..
    java -jar gui/CodeLabellingGUI.jar & python3 cesres_graphcodebert_model.py -c gui/config_gui.yaml
    cd gui
else
    echo "Please run this script when inside the directory containing it."
fi