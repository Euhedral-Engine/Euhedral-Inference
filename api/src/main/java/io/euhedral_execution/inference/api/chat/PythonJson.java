package io.euhedral_execution.inference.api.chat;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.MathContext;
import java.math.RoundingMode;
import java.util.List;
import java.util.Map;

/// Serializes JSON exactly as the checkpoint template's `tojson` filter, which Hugging Face defines as
/// Python `json.dumps(value, ensure_ascii=False)`: `", "` and `": "` separators, insertion key order,
/// non-ASCII text unescaped, and floats in Python `repr` form (`1e-05`, `1e+16`, `2.0`).
///
/// Accepts the untyped values Jackson produces for JSON: maps with string keys, lists, strings, booleans,
/// null, `Integer`/`Long`/`BigInteger`, and `Double`/`BigDecimal`. Non-finite doubles use Python's
/// `allow_nan` spellings `Infinity`, `-Infinity`, and `NaN`.
final class PythonJson {

    private PythonJson() {}

    static String dumps(Object value) {
        StringBuilder json = new StringBuilder();
        write(value, json);
        return json.toString();
    }

    private static void write(Object value, StringBuilder json) {
        switch (value) {
            case null -> json.append("null");
            case String text -> writeString(text, json);
            case Boolean bool -> json.append(bool ? "true" : "false");
            case Integer _, Long _, Short _, Byte _, BigInteger _ -> json.append(value);
            case Double number -> json.append(pythonFloat(number));
            case Float number -> json.append(pythonFloat(number.doubleValue()));
            // Python parses every JSON fraction as a float.
            case BigDecimal number -> json.append(pythonFloat(number.doubleValue()));
            case Map<?, ?> map -> {
                json.append('{');
                boolean first = true;
                for (var entry : map.entrySet()) {
                    if (!(entry.getKey() instanceof String key))
                        throw new IllegalArgumentException("JSON object keys must be strings");
                    if (!first) json.append(", ");
                    first = false;
                    writeString(key, json);
                    json.append(": ");
                    write(entry.getValue(), json);
                }
                json.append('}');
            }
            case List<?> list -> {
                json.append('[');
                for (int index = 0; index < list.size(); index++) {
                    if (index > 0) json.append(", ");
                    write(list.get(index), json);
                }
                json.append(']');
            }
            default ->
                throw new IllegalArgumentException(
                        "not a JSON value: " + value.getClass().getName());
        }
    }

    /// Python escapes only `"`, `\`, and C0 controls (four lowercase hex digits except the short escapes).
    private static void writeString(String text, StringBuilder json) {
        json.append('"');
        for (int index = 0; index < text.length(); index++) {
            char value = text.charAt(index);
            switch (value) {
                case '"' -> json.append("\\\"");
                case '\\' -> json.append("\\\\");
                case '\n' -> json.append("\\n");
                case '\r' -> json.append("\\r");
                case '\t' -> json.append("\\t");
                case '\b' -> json.append("\\b");
                case '\f' -> json.append("\\f");
                default -> {
                    if (value < 0x20) json.append(String.format("\\u%04x", (int) value));
                    else json.append(value);
                }
            }
        }
        json.append('"');
    }

    /// Python `float.__repr__`: the shortest round-tripping digits, positional when the decimal exponent is
    /// in [-4, 16), otherwise `d.ddde±XX`.
    static String pythonFloat(double value) {
        if (Double.isNaN(value)) return "NaN";
        if (Double.isInfinite(value)) return value > 0 ? "Infinity" : "-Infinity";
        if (value == 0.0) return Double.doubleToRawLongBits(value) < 0 ? "-0.0" : "0.0";
        BigDecimal decimal = new BigDecimal(Double.toString(value)).stripTrailingZeros();
        // Double.toString is shortest except that it prints two digits when one suffices (4.9E-324, not 5e-324).
        if (decimal.precision() == 2) {
            BigDecimal oneDigit = new BigDecimal(value).round(new MathContext(1, RoundingMode.HALF_EVEN));
            if (oneDigit.doubleValue() == value) decimal = oneDigit.stripTrailingZeros();
        }
        String digits = decimal.unscaledValue().abs().toString();
        int exponent = digits.length() - 1 - decimal.scale();
        String sign = value < 0 ? "-" : "";
        if (exponent >= -4 && exponent < 16) {
            String plain = decimal.abs().toPlainString();
            return sign + (plain.indexOf('.') < 0 ? plain + ".0" : plain);
        }
        String mantissa = digits.length() == 1 ? digits : digits.charAt(0) + "." + digits.substring(1);
        String exponentDigits = Integer.toString(Math.abs(exponent));
        if (exponentDigits.length() == 1) exponentDigits = "0" + exponentDigits;
        return sign + mantissa + "e" + (exponent < 0 ? "-" : "+") + exponentDigits;
    }
}
