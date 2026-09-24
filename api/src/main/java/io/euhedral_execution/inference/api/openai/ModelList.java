package io.euhedral_execution.inference.api.openai;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

/// `/v1/models` listing. Only the public model ID is exposed, never artifact paths or topology.
public record ModelList(String object, List<Model> data) {

    public static ModelList of(Model model) {
        return new ModelList("list", List.of(model));
    }

    public record Model(
            String id,
            String object,
            long created,
            @JsonProperty("owned_by") String ownedBy) {

        public static Model of(String id, long created) {
            return new Model(id, "model", created, "euhedral");
        }
    }
}
