from datetime import datetime


def open_setup_file():
    with open("setup.py", "r") as f:
        file_lines = f.readlines()
    return file_lines


def replace_nightly_package_info(file_lines):
    package_name = "colossalai-nightly"

    for idx, line in enumerate(file_lines):
        if 'package_name = "colossalai"' in line:
            file_lines[idx] = f'package_name = "{package_name}"\n'
    return file_lines


def write_setup_file(file_lines):
    with open("setup.py", "w") as f:
        f.writelines(file_lines)


def main():
    file_lines = open_setup_file()
    file_lines = replace_nightly_package_info(file_lines)
    write_setup_file(file_lines)
    # Keep setup.py's get_version() call so the runtime version module is generated.
    with open("version.txt", "w") as f:
        f.write(datetime.today().strftime("%Y.%m.%d") + "\n")


if __name__ == "__main__":
    main()
