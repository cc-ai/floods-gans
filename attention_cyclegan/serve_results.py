import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import numpy as np


def parametrized_web_server(loc):
    class Web_server(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self.path = "/epoch/latest"

            if self.path[-1] != "/":
                self.path += "/"

            if "epoch" not in self.path:
                file_to_open = "Nothing here. Try /epoch/latest or /epoch/latest"
                self.send_response(404)
            else:
                epoch_files = list(loc.glob("*epoch*.html"))
                if "latest" in self.path:
                    epoch_indexes = [
                        int(e.split("_")[1].split(".")[0]) for e in epoch_files
                    ]
                    file_to_open = str(epoch_files[np.argmax(epoch_indexes)])
                else:
                    file_to_open = str(epoch_files[int(self.path.split("/")[-2])])
                try:
                    # Reading the file
                    file_to_open = open(file_to_open).read()
                    self.send_response(200)
                except FileNotFoundError:
                    file_to_open = "File not found. Available epochs:\n" + "\n".join(
                        str(e) for e in epoch_files
                    )
                    self.send_response(404)

            self.end_headers()
            self.wfile.write(bytes(file_to_open, "utf-8"))

    return Web_server


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-f",
        "--files",
        required=False,
        help="where the epoch files are located from current working directory",
    )
    args = vars(ap.parse_args())
    loc = Path(args["files"]).resolve()

    if len(list(loc.glob("*epoch*.html"))) == 0:
        raise AssertionError(
            "No epoch files found in"
            + str(loc)
            + "\n".join(str(l) for l in loc.glob("*epoch*.html"))
        )

    with HTTPServer(("localhost", 8080), parametrized_web_server(loc)) as httpd:
        httpd.serve_forever()
