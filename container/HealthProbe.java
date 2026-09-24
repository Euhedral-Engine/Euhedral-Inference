import java.net.HttpURLConnection;
import java.net.URI;

/// Container health probe; uses only the Java runtime, not a shell or curl.
public final class HealthProbe {
    public static void main(String[] args) throws Exception {
        int port = Integer.parseInt(System.getenv().getOrDefault("PORT", "1738"));
        HttpURLConnection connection = (HttpURLConnection) URI.create("http://127.0.0.1:" + port + "/health")
                .toURL()
                .openConnection();
        connection.setConnectTimeout(2000);
        connection.setReadTimeout(2000);
        try {
            if (connection.getResponseCode() != 200) System.exit(1);
        } finally {
            connection.disconnect();
        }
    }
}
