# Using CodeLabellingGUI with CESReS

Author: Guillaume Steveny; 
Year: 2023 - 2024

This file explains how to run the model with the GUI we designed.

We created two scripts to launch the two programs:
- `launch_gui.bat` for Windows
- `launch_gui.sh` for Linux

Using these will start the CodeLabellingGUI.jar program and the python model.
Be sure to use them when inside the directory containing these.

When the model is ready to accept connection, "Started serving" appears.
You can click on the "Model" menu and "Connect to model".
If the configuration works, you will get a success message.

You can write code on the left panel and use send code to ask the model for its predictions.
Once the list appears in the right panel, you can click on a label to ask the model for its interpretability results.
After some time depending on your computer power, you will get color on the left panel indicating positive and negative contributions.
Values close to 1 are darker shades of green, values close to -1 are darker shades of red.
Clicking inside the code panel removes the highlights.

Inside the "Model" menu you can disconnect from the model or directly close the app.
To close the Python server, you will need to send a SIGINT to the program.
Doing CTRL+C will perform it.

The code source will be available under the repository address:
https://github.com/StevenGuyCap/CodeLabellingGUI