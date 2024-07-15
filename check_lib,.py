import pkg_resources

# Danh sách các thư viện bạn muốn kiểm tra
libraries_to_check = ["numpy", "pandas", "tensorflow"]

for library_name in libraries_to_check:
    try:
        version = pkg_resources.get_distribution(library_name).version
        print(f"{library_name} version: {version}")
    except pkg_resources.DistributionNotFound:
        print(f"{library_name} is not installed.")
