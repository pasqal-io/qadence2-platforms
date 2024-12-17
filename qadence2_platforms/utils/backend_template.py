from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from shutil import copyfile

from qadence2_platforms import CUSTOM_BACKEND_FOLDER_NAME, TEMPLATES_FOLDER_NAME
from qadence2_platforms.utils.module_importer import resolve_module_path

try:
    from tkinter import filedialog as fd

except ImportError:

    def user_input() -> str:
        return input("Paste the directory: ")

else:

    def user_input() -> str:
        return fd.askdirectory(
            initialdir=Path(__file__).parent,
            title="Select a directory",
            mustexist=True,
        )


class BackendTemplate:
    """
    Class to create new custom backend folder with file templates.

    It will
    follow the same structure as the built-in backends, namely `fresnel1` and
    `pyqtorch`, having a folder with the custom backend name with the essential
    files inside already built with the functions, classes and their methods.

    Ex:

    ```
    selected_root_dir/
    └── custom_backends/
        ├── custom_backend1/
        │   ├── __init__.py
        │   ├── compiler.py
        │   └── interface.py
    ```

    It is intended to help the user when creating their custom backend with
    pre-filled files so the core structure of the backend instances are still
    present, while giving freedom for them to implement whatever else needed.
    """

    def __init__(self) -> None:
        self._pwd: Path = Path()
        self._root_backends_name: str = CUSTOM_BACKEND_FOLDER_NAME
        self._backend_name: str = ""
        self._backend_path: Path = Path()
        self._template_path: Path = self._get_template_path()
        self._template_files_list: list[str] = os.listdir(self._template_path)

    @property
    def user_backend_path(self) -> Path:
        return self._backend_path

    @property
    def platforms_backend_path(self) -> Path:
        return Path()

    @property
    def template_files_list(self) -> list[str]:
        return self._template_files_list

    def _new_file_path(self, name: str) -> Path:
        return self._backend_path / name

    def _get_template_path(self) -> Path:
        return Path(__file__).parent / TEMPLATES_FOLDER_NAME

    def create_folder(self, backend_name: str, current_path: str | Path) -> bool:
        """
        Creates the main folder for the custom backend in a selected path.

        Args:
            backend_name (str): backend name
            current_path (str, Path): current path to place the custom backend

        Returns:
            Returns true if the folder was already existing
        """

        self._backend_name = backend_name
        self._pwd = Path(current_path)
        self._backend_path = Path(self._pwd, self._root_backends_name, self._backend_name)
        already_exists = os.path.exists(self._backend_path)
        self._backend_path.mkdir(parents=True, exist_ok=True)
        return already_exists

    def create_files(self) -> None:
        for file in self.template_files_list:
            self._new_file_path(file).touch()
            copyfile(self._template_path / file, self._backend_path / file)

    def create_template(
        self,
        backend_name: str,
        gui: bool = True,
        use_this_dir: str | Path | None = None,
    ) -> bool:
        """
        Creates the template, with the main custom backend folder and its content files.

        Args:
            backend_name (str): backend name
            gui (bool): whether to use a GUI option to choose where to create the template;
                tkinter must be installed.
                On Mac: `brew install python-tk@python3.10`. In case you use a different python
                version, replace `3.10` by it. On Linux: `apt-get install python-tk`.
            use_this_dir (str | Path | None): directory to create the custom backend

        Returns:
             Returns true if the template was successfully created
        """

        print(
            "\nCreating a backend template.\n\n"
            "You need to select a directory where all the custom backends will be located.\n"
            " The organization will be as follows:\n\n"
            "   selected_root_dir/\n"
            "   └── custom_backends/\n"
            "       ├── custom_backend1/\n"
            "       │   ├── __init__.py\n"
            "       │   ├── compiler.py\n"
            "       │   └── interface.py\n"
            "       ├── custom_backend2/\n"
            "       ...\n"
            ""
        )

        selected_dir: str | Path

        if gui and use_this_dir is None and "tkinter" in sys.modules:
            selected_dir = user_input()
        else:
            if use_this_dir is None:
                selected_dir = user_input()
            else:
                selected_dir = use_this_dir

        if selected_dir:
            try:
                already_exists = self.create_folder(backend_name, current_path=selected_dir)
                self.create_files()
            except Exception:
                traceback.print_exc()
                return False
            else:
                action = "replaced" if already_exists else "created"
                result = resolve_module_path(Path(selected_dir, CUSTOM_BACKEND_FOLDER_NAME))
                if result:
                    print(
                        f"Backend template at {self._backend_path} has been {action} with success!"
                    )
                return result
        else:
            print("You must select a directory to create the template. Creation suspended.")
            return False
