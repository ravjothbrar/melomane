"""Test-session environment hygiene."""
import os

# Some IDEs / agents / debugging proxies export SSLKEYLOGFILE.  On the python.org / uv Windows builds, Python's ssl module then
# calls into OpenSSL's fopen shim and the whole interpreter dies with "OPENSSL_Uplink: no OPENSSL_Applink" on the first
# urllib/httpx request - which under pytest shows up as a silent hang in any test that opens a local socket.
os.environ.pop("SSLKEYLOGFILE", None)
