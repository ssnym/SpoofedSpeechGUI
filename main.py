import matplotlib
matplotlib.use('Qt5Agg') # Explicitly set the backend for PyQt

import librosa
import librosa.display
import numpy as np
import os, glob
import warnings
from datetime import datetime # Import the datetime module

from main_aasist import aasist_model
from main_rawnet import rawnet_model

import matplotlib.pyplot as plt

from PyQt6.QtWidgets import QMainWindow, QApplication, QLabel, QGridLayout, QPushButton, QFileDialog, QWidget, QHBoxLayout, QDialog, QTextEdit, QVBoxLayout, QTableWidget, QTableWidgetItem
from PyQt6.QtCore import Qt , QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

warnings.filterwarnings("ignore", category=FutureWarning)




class ResultsDialog(QDialog):
    """A dialog to display test results in a proper table."""
    def __init__(self, results_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Test Results")
        self.setMinimumSize(950, 400)
        self.results_data = results_data

        # --- Layout and Widgets ---
        layout = QVBoxLayout(self)

        # Create the table widget
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers) # Make table read-only
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)

        self._populate_table() # Fill the table with data

        self.save_button = QPushButton("Save Results as CSV")
        self.save_button.setFixedSize(150, 40)
        self.save_button.clicked.connect(self.save_results)
        layout.addWidget(self.save_button, alignment=Qt.AlignmentFlag.AlignRight)

        self.setLayout(layout)

    def _populate_table(self):
        """Fills the QTableWidget with the results data."""
        if not self.results_data:
            return

        headers = ["Filename", "prob of spoof (AASIST) (%)", "prob of spoof (RawNet) (%)", "prob of spoof (One-Class) (%)", "Final prob of spoof (%)"]
        self.table.setRowCount(len(self.results_data))
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)

        for row_idx, item_data in enumerate(self.results_data):
            # Create QTableWidgetItem for each piece of data
            filename = QTableWidgetItem(item_data['filename'])
            a_spoof_confidence = QTableWidgetItem(f"{item_data['a_spoof_confidence']*100:.2f}")
            r_spoof_confidence = QTableWidgetItem(f"{item_data['r_spoof_confidence']*100:.2f}")
            # oc_spoof_confidence and oc_result are placeholders as in the original code
            oc_spoof_confidence = QTableWidgetItem('N/A') # Set to N/A as per original code for display
            final_score = QTableWidgetItem(f"{item_data['final_score']*100:.2f}")

            # Center-align the scores and results for better readability
            # Note: oc_spoof_confidence and final_result are also included for alignment,
            #       even though oc_spoof_confidence might be 'N/A'.
            for cell in [filename, a_spoof_confidence, r_spoof_confidence, oc_spoof_confidence, final_score]:
                cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            # Place items into the table
            self.table.setItem(row_idx, 0, filename)
            self.table.setItem(row_idx, 1, a_spoof_confidence)
            self.table.setItem(row_idx, 2, r_spoof_confidence)
            self.table.setItem(row_idx, 3, oc_spoof_confidence)
            self.table.setItem(row_idx, 4, final_score)

        # Automatically resize columns to fit the content
        self.table.resizeColumnsToContents()

    def save_results(self):
        """Opens a file dialog to save the results as a CSV file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            try:
                with open(file_path, 'w', newline='') as f: # Added newline='' for proper CSV writing
                    # Get current date and time
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"Results generated on: {current_time}\n\n") # Write date and time, followed by an extra newline for spacing

                    # Write header
                    headers = [self.table.horizontalHeaderItem(i).text() for i in range(self.table.columnCount())]
                    f.write(",".join(headers) + "\n")
                    # Write data rows
                    for row in range(self.table.rowCount()):
                        row_data = []
                        for col in range(self.table.columnCount()):
                            item = self.table.item(row, col)
                            if item is not None:
                                row_data.append(item.text())
                            else:
                                row_data.append('') # Handle empty cells
                        f.write(",".join(row_data) + "\n")
            except Exception as e:
                print(f"Error saving file: {e}")

     
        

class Window(QMainWindow):
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.setWindowTitle('Audio Deepfake Test')
        self.resize(900, 500)
        
        parent_layout = QGridLayout()
        
        self.audio_path = None
        self.audio_folder=None
        self.audio_folder_files = []
        
        # Set up the player and audio output
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.player.mediaStatusChanged.connect(self.media_status_changed_Handler)

        
        # Top Bar
        self.open_btn = QPushButton('File')
        self.open_btn.setFixedSize(100,40)
        
        self.open_btn.clicked.connect(self.open_btn_Handler)
        
        # ---------------------------------------------------------------------------------
        self.center_top_bar = QHBoxLayout()
        
        self.open_folder_btn=QPushButton('Folder')
        self.open_folder_btn.setFixedSize(120, 40)
        
        self.open_folder_btn.clicked.connect(self.open_folder_btn_Handler)
        
        self.file_label = QLabel('Select File via Open')
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.audio_btn = QPushButton('Play')
        self.audio_btn.setFixedSize(100,40)        
        self.audio_btn.clicked.connect(self.audio_btn_Handler)
        self.audio_btn.setEnabled(False) # Disabled by default
        
        self.center_top_bar.addWidget(self.open_folder_btn)
        self.center_top_bar.addWidget(self.file_label)
        self.center_top_bar.addWidget(self.audio_btn)
        # ---------------------------------------------------------------------

        self.test_btn = QPushButton('Test')
        self.test_btn.setFixedSize(100,40)
        self.test_btn.clicked.connect(self.test_btn_Handler)
        self.test_btn.setEnabled(False) # Disabled by default

        # Mel - Spectrogram Bar
        self.mel_spec_label = QLabel('Spectrogram')
        self.mel_spec_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fig_spec , self.ax_spec = plt.subplots()
        self.mel_spec_canvas = FigureCanvas(self.fig_spec)
        
        # Waveform Bar
        self.waveform_label=QLabel('Audio Waveform')
        self.waveform_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fig_waveform, self.ax_waveform = plt.subplots()
        self.waveform_canvas = FigureCanvas(self.fig_waveform)
        
        # Result Bar
        result_layout = QHBoxLayout()
        self.aasist_label = QLabel('prob of spoof (AASIST):  click Test')
        self.rawnet_label = QLabel('prob of spoof (Raw-Net):  click Test')
        self.one_class_label = QLabel('prob of spoof (One-Class):  click Test')
        self.rawnet_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.one_class_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        result_layout.addWidget(self.aasist_label)
        result_layout.addWidget(self.rawnet_label)
        result_layout.addWidget(self.one_class_label)
        
        # Last Bar
        self.final_result_label=QLabel('Final prob of spoof : None')
        self.final_result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        
        # Top Bar
        parent_layout.addWidget(self.open_btn, 0, 0)
        parent_layout.addLayout(self.center_top_bar, 0, 1)
        parent_layout.addWidget(self.test_btn, 0, 2)
        # Mel - Spectrogram Bar
        parent_layout.addWidget(self.mel_spec_label, 1, 0)
        parent_layout.addWidget(self.mel_spec_canvas, 1, 1, 1, 2)
        # Waveform Bar
        parent_layout.addWidget(self.waveform_label, 2, 0)
        parent_layout.addWidget(self.waveform_canvas, 2, 1, 1, 2)
        # Result Bar
        parent_layout.addLayout(result_layout, 3, 0, 1, 3)
        # Last Bar
        parent_layout.addWidget(self.final_result_label, 4, 0, 1, 3)
        
        # Set Layout
        center_widget=QWidget()
        center_widget.setLayout(parent_layout)
        self.setCentralWidget(center_widget)
        
       
    def open_btn_Handler(self):
        
        dialog=QFileDialog()
        dialog.setWindowTitle("Select File")
        dialog.setNameFilter('Audio Files (*.flac *.mp3 *.wav)')
        dialog_return = dialog.exec()
        
        self.audio_path = dialog.selectedFiles()[0]
        
        if dialog_return:
            self.file_label.setText(self.audio_path.split('/')[-1])
            self.audio_folder=None
            self.audio_folder_files=[]
            self.audio_btn.setText("Play")
            self.audio_btn.setEnabled(True)
            self.test_btn.setEnabled(True)
            self.display_audio_Handler()
            
            
    def open_folder_btn_Handler(self):
        
        dialog = QFileDialog(self)
        dialog.setWindowTitle("Select Folder")
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)

        if dialog.exec():
            self.audio_folder = dialog.selectedFiles()[0]
            extensions=('*.mp3', '*.wav', '*.flac')
            for ext in extensions:
                self.audio_folder_files.extend(glob.glob(os.path.join(self.audio_folder, ext)))
            if not self.audio_folder_files:
                self.file_label.setText('No Audio file found in folder')
            else:
                self.file_label.setText(f"{os.path.basename(self.audio_folder)} ({len(self.audio_folder_files)} files)")
                self.audio_path = None
                self.audio_btn.setEnabled(False)
                self.test_btn.setEnabled(True)

            
    
    def audio_btn_Handler(self):
        
        if self.audio_path is None:
            return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.audio_btn.setText("Play")
        else:
            self.player.play()
            self.audio_btn.setText("Pause")
            
            
    def test_btn_Handler(self):
        

        files_to_process = []
        if self.audio_path:
            # files_to_process.append(self.audio_path) # No need to append to this list if processing single file immediately
            a_spoof_confidence, a_result = aasist_model(self.audio_path)
            r_spoof_confidence, r_result = rawnet_model(self.audio_path)
            # oc_spoof_confidence and oc_result are placeholders as in the original code
            oc_spoof_confidence, oc_result = 0, 0 # Keep as 0, 0 for now as per current logic
            
            # Determine final result by majority vote (from the 3 models - assuming oc_result is 0)
            final_result_bool = (a_result + r_result + oc_result) >= 2
            # Average score of the two active models
            final_spoof_confidence = (a_spoof_confidence + r_spoof_confidence) / 2
            
            self.aasist_label.setText(f'prob of spoof (AASIST): {a_spoof_confidence*100:.2f} ')
            self.rawnet_label.setText(f'prob of spoof (RawNet): {r_spoof_confidence*100:.2f} ')
            self.one_class_label.setText(f'prob of spoof (One-Class): N/A')
            self.final_result_label.setText(f'Final prob of spoof : {final_spoof_confidence*100:.2f} % ') # Corrected display
            
            # For single file, still show in dialog for consistency if desired, otherwise remove.
            # Currently, single file results are displayed in labels, not the dialog.
            return
            
        elif self.audio_folder_files:
            files_to_process.extend(self.audio_folder_files)
            # Reset labels when processing a folder
            self.aasist_label.setText(f'prob of spoof (AASIST): Processing...')
            self.rawnet_label.setText(f'prob of spoof (RawNet): Processing...')
            self.one_class_label.setText(f'prob of spoof (One-Class): Processing...')
            self.final_result_label.setText(f'Final prob of spoof : Processing folder...')
        else:
            self.final_result_label.setText("Please select a file or folder first.")
            return

        all_results_data = []
        for file_path in files_to_process:
            a_spoof_confidence, a_result_raw = aasist_model(file_path)
            r_spoof_confidence, r_result_raw = rawnet_model(file_path)
            
            # Assume a_result_raw and r_result_raw are 0 for bonafide, 1 for spoofed (or vice versa, check your model's output)
            # For a_result and r_result in the table, we want "Bonafide" or "Spoofed" string.
            # Let's assume 0 means bonafide and 1 means spoofed from your model's a_result and r_result.
            # If your model outputs 0 for spoofed and 1 for bonafide, you'd reverse these.
            
            # oc_spoof_confidence and oc_result are placeholders
            oc_spoof_confidence, oc_result_raw = 0, 0 

            # Determine final result by majority vote (from the 3 models)
            # Ensure consistency: if 0 is Bonafide and 1 is Spoofed:
            # We want to count how many models say "spoofed"
            # spoofed_count = 0
            # if a_result_raw == 1: # Assuming 1 means spoofed
            #     spoofed_count += 1
            # if r_result_raw == 1:
            #     spoofed_count += 1
            # if oc_result_raw == 1:
            #     spoofed_count += 1

            # final_result_bool = (spoofed_count >= 2) # True if 2 or more models say spoofed

            # Average score of the two active models
            final_score = (a_spoof_confidence + r_spoof_confidence) / 2
            # final_result_str = "Spoofed" if final_result_bool else "Bonafide"

            # Store data for the results dialog
            all_results_data.append({
                "filename": os.path.basename(file_path),
                "a_spoof_confidence": a_spoof_confidence, 
                "r_spoof_confidence": r_spoof_confidence, 
                "oc_spoof_confidence": oc_spoof_confidence, # Keep this even if N/A for consistency
                "final_score": final_score, 
            })
            
        if all_results_data: # Check if there is data to display
            dialog = ResultsDialog(all_results_data, self) 
            dialog.exec()
  
    # This method is no longer used since ResultsDialog now handles table display directly.
    # It can be removed or kept for reference if text display logic is needed elsewhere.
    def _format_results_for_display(self, results_data):
        """
        Creates a neatly formatted, aligned string from the results data for display.
        The column width for the filename is calculated dynamically.
        """
        if not results_data:
            return "No audio files were processed."

        filename_col_width = max(len(item['filename']) for item in results_data) + 3
        filename_col_width = max(filename_col_width, len('Filename') + 3)

        score_col_width = 12
        res_col_width = 12

        header = (
            f"{'Filename':<{filename_col_width}} | "
            f"{'AASIST (%)':<{score_col_width}} | {'AASIST Res':<{res_col_width}} | "
            f"{'RawNet (%)':<{score_col_width}} | {'RawNet Res':<{res_col_width}} | "
            f"{'Final (%)':<{score_col_width}} | {'Final Res'}\n"
        )

        total_width = (filename_col_width + (score_col_width * 3) + (res_col_width * 2) + (3 * 5))
        separator = "-" * total_width + "\n"

        body = ""
        for item in results_data:
            body += (
                f"{item['filename']:<{filename_col_width}} | "
                f"{item['a_spoof_confidence']*100:<{score_col_width}.2f} | {item['a_result']:<{res_col_width}} | "
                f"{item['r_spoof_confidence']*100:<{score_col_width}.2f} | {item['r_result']:<{res_col_width}} | "
                f"{item['final_score']*100:<{score_col_width}.2f} | {item['final_result']}\n"
            )

        return header + separator + body
                
            
            
    
    def display_audio_Handler(self):
        
        audio_data, sample_rate = librosa.load(self.audio_path)
        
        # Mel Spectrogram
        self.ax_spec.clear()
        S = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, sr=sample_rate, x_axis='time', y_axis='mel', cmap='viridis', ax=self.ax_spec)
        # self.ax_spec.colorbar(format='%+2.0f dB')
        self.ax_spec.set_title('Mel-Spectrogram')
        self.mel_spec_canvas.draw()
        
        # Audio Waveform
        self.ax_waveform.clear()
        librosa.display.waveshow(y=audio_data, sr=sample_rate, axis='time', ax=self.ax_waveform)
        self.ax_waveform.set_title("Waveform")
        self.waveform_canvas.draw()
        
        self.player.setSource(QUrl.fromLocalFile(self.audio_path))
        

    def media_status_changed_Handler(self, status):
        
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.audio_btn.setText("Play")

            
        

def main():
    
    app = QApplication([])
    window=Window()
    window.show()
    app.exec()
    
if __name__ == "__main__":
    main()