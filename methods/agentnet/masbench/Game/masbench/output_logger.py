"""
Terminal output redirection and persistence utilities.
Display all output in the terminal while saving it to a file.
"""

import sys
import os
import threading
from datetime import datetime
from typing import Optional, TextIO


class TeeOutput:
    """
    Redirect output to both terminal and file.
    Supports thread-safe file writes.
    """

    def __init__(self, file_path: str, original_stream: TextIO):
        """
        Initialize output redirection.

        Args:
            file_path: Log file path.
            original_stream: Original output stream, usually sys.stdout.
        """
        print(f"DEBUG: [OUTPUT_LOGGER] Initializing output redirection to: {file_path}")

        self.file_path = file_path
        self.original_stream = original_stream
        # Preserve the real terminal stream for correct color output.
        self.real_terminal = sys.__stdout__  # Actual terminal output stream.
        self.file_stream = None
        self.lock = threading.Lock()

        # Ensure the directory exists.
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Open the file for writing.
        try:
            self.file_stream = open(file_path, 'w', encoding='utf-8', buffering=1)  # Line buffering.
            print(f"DEBUG: [OUTPUT_LOGGER] Opened log file: {file_path}")

            # Write the file header.
            header = f"=== MASBench Terminal Output Log ===\n"
            header += f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            header += f"Log file: {file_path}\n"
            header += "=" * 50 + "\n\n"
            self.file_stream.write(header)
            self.file_stream.flush()

        except Exception as e:
            print(f"ERROR: [OUTPUT_LOGGER] Cannot open log file {file_path}: {e}")
            self.file_stream = None

    def _remove_ansi_codes(self, text: str) -> str:
        """
        Remove ANSI escape sequences (color codes).

        Args:
            text: Text containing ANSI codes.

        Returns:
            Plain text with ANSI codes removed.
        """
        import re
        # Comprehensive ANSI escape sequence patterns.
        # Match colors, cursor controls, styles, and other ANSI sequences.
        ansi_patterns = [
            r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])',  # Standard ANSI sequences.
            r'\x1B\[[0-9;]*[mGKHf]',                     # Common color and cursor sequences.
            r'\x1B\[[0-9;]*[ABCDEFGJKSTf]',             # Cursor movement sequences.
            r'\x1B\].*?\x07',                            # OSC sequences ending with BEL.
            r'\x1B\].*?\x1B\\',                          # OSC sequences ending with ST.
        ]

        clean_text = text
        for pattern in ansi_patterns:
            ansi_escape = re.compile(pattern)
            clean_text = ansi_escape.sub('', clean_text)

        return clean_text

    def write(self, text: str) -> None:
        """
        Write text to the terminal and file.

        Args:
            text: Text to write.
        """
        with self.lock:
            # Prefer the real terminal for correct color display.
            if self.real_terminal:
                try:
                    self.real_terminal.write(text)
                    self.real_terminal.flush()
                except Exception as e:
                    # Fall back to the original stream if the real terminal write fails.
                    if self.original_stream:
                        try:
                            self.original_stream.write(text)
                            self.original_stream.flush()
                        except Exception:
                            pass

            # Strip ANSI color codes before writing to the file.
            if self.file_stream:
                try:
                    clean_text = self._remove_ansi_codes(text)
                    self.file_stream.write(clean_text)
                    self.file_stream.flush()  # Write to disk immediately.
                except Exception as e:
                    # Report file write failures to the real terminal.
                    if self.real_terminal:
                        self.real_terminal.write(f"ERROR: [OUTPUT_LOGGER] File write failed: {e}\n")

    def flush(self) -> None:
        """Flush all output streams."""
        with self.lock:
            # Flush the real terminal first.
            if self.real_terminal:
                try:
                    self.real_terminal.flush()
                except Exception:
                    pass

            if self.original_stream:
                try:
                    self.original_stream.flush()
                except Exception:
                    pass

            if self.file_stream:
                try:
                    self.file_stream.flush()
                except Exception:
                    pass

    def close(self) -> None:
        """Close the file stream."""
        print(f"DEBUG: [OUTPUT_LOGGER] Closing output redirection")

        with self.lock:
            if self.file_stream:
                try:
                    # Write the file footer.
                    footer = f"\n\n" + "=" * 50 + "\n"
                    footer += f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    footer += "=== End of MASBench Terminal Output Log ===\n"
                    self.file_stream.write(footer)
                    self.file_stream.flush()
                    self.file_stream.close()
                    print(f"DEBUG: [OUTPUT_LOGGER] Log file closed: {self.file_path}")
                except Exception as e:
                    print(f"ERROR: [OUTPUT_LOGGER] Failed to close log file: {e}")
                finally:
                    self.file_stream = None


class OutputManager:
    """
    Manage the lifecycle of output redirection.
    """

    def __init__(self, log_dir: str = "txt_logs", experiment_name: Optional[str] = None):
        """
        Initialize the output manager.

        Args:
            log_dir: Log directory.
            experiment_name: Optional experiment name.
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.tee_output = None
        self.original_stdout = None
        self.is_redirected = False

        print(f"DEBUG: [OUTPUT_MANAGER] Initializing output manager; log directory: {log_dir}")

    def start_logging(self) -> str:
        """
        Start logging by redirecting stdout.

        Returns:
            Log file path.
        """
        if self.is_redirected:
            print(f"WARNING: [OUTPUT_MANAGER] Output redirection is already active")
            return None

        # Generate the log file path.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.experiment_name:
            filename = f"terminal_output_{self.experiment_name}_{timestamp}.txt"
        else:
            filename = f"terminal_output_{timestamp}.txt"

        log_file_path = os.path.join(self.log_dir, filename)

        print(f"DEBUG: [OUTPUT_MANAGER] Starting terminal output redirection to: {log_file_path}")

        # Save the original stdout.
        self.original_stdout = sys.stdout

        # Create TeeOutput and redirect.
        self.tee_output = TeeOutput(log_file_path, self.original_stdout)
        sys.stdout = self.tee_output

        self.is_redirected = True
        print(f"DEBUG: [OUTPUT_MANAGER] Terminal output redirection started")

        return log_file_path

    def stop_logging(self) -> None:
        """
        Stop logging and restore stdout.
        """
        if not self.is_redirected:
            print(f"DEBUG: [OUTPUT_MANAGER] Output redirection is inactive; nothing to stop")
            return

        print(f"DEBUG: [OUTPUT_MANAGER] Stopping terminal output redirection")

        # Restore original stdout.
        if self.original_stdout:
            sys.stdout = self.original_stdout

        # Close TeeOutput.
        if self.tee_output:
            self.tee_output.close()
            self.tee_output = None

        self.is_redirected = False
        print(f"DEBUG: [OUTPUT_MANAGER] Terminal output redirection stopped")

    def __enter__(self):
        """Enter the context manager."""
        return self.start_logging()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        self.stop_logging()


# Global output manager instance.
_global_output_manager: Optional[OutputManager] = None


def start_global_logging(log_dir: str = "txt_logs", experiment_name: Optional[str] = None) -> str:
    """
    Start global output logging.

    Args:
        log_dir: Log directory.
        experiment_name: Experiment name.

    Returns:
        Log file path.
    """
    global _global_output_manager

    if _global_output_manager and _global_output_manager.is_redirected:
        print(f"WARNING: Global output logging is already active")
        return None

    _global_output_manager = OutputManager(log_dir, experiment_name)
    return _global_output_manager.start_logging()


def stop_global_logging() -> None:
    """Stop global output logging."""
    global _global_output_manager

    if _global_output_manager:
        _global_output_manager.stop_logging()
        _global_output_manager = None


# Test code.
if __name__ == "__main__":
    print("=== Testing OutputLogger ===")

    # Test 1: context manager.
    print("\n--- Test 1: Context manager mode ---")
    with OutputManager("test_logs", "test_experiment") as log_file:
        print(f"Log file: {log_file}")
        print("Test output 1")
        print("Test output 2")
        print("Sample text output: Hello, world!")

    print("Context manager test complete")

    # Test 2: manual management.
    print("\n--- Test 2: Manual management mode ---")
    manager = OutputManager("test_logs", "manual_test")
    log_file = manager.start_logging()
    print(f"Manual mode log file: {log_file}")
    print("Manual mode test output 1")
    print("Manual mode test output 2")
    manager.stop_logging()

    print("Manual management test complete")

    # Test 3: global mode.
    print("\n--- Test 3: Global mode ---")
    log_file = start_global_logging("test_logs", "global_test")
    print(f"Global mode log file: {log_file}")
    print("Global mode test output 1")
    print("Global mode test output 2")
    stop_global_logging()

    print("Global mode test complete")
    print("=== OutputLogger tests complete ===")
