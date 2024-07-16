use std::collections::{HashMap, HashSet};
use serde_json::json;

fn get_last_key_part(key: &str) -> &str {
    key.split('.').last().unwrap_or(key)
}

pub struct Shapeshift {
    embedding_client: String,
    api_key: String,
    embedding_model: String,
    similarity_threshold: f64,
}

impl Shapeshift {
    pub fn new(embedding_client: String, api_key: String, embedding_model: String, similarity_threshold: f64) -> Self {
        Self {
            embedding_client,
            api_key,
            embedding_model,
            similarity_threshold,
        }
    }

    async fn calculate_embeddings(&self, texts: Vec<String>) -> Vec<Vec<f64>> {
        get_embeddings(&texts)
    }

    fn cosine_similarity(&self, vec1: &[f64], vec2: &[f64]) -> f64 {
        let dot_product: f64 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
        let magnitude1: f64 = vec1.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let magnitude2: f64 = vec2.iter().map(|&x| x * x).sum::<f64>().sqrt();
        dot_product / (magnitude1 * magnitude2)
    }

    fn find_closest_match(&self, source_embedding: &[f64], target_embeddings: &[Vec<f64>]) -> Option<usize> {
        target_embeddings
            .iter()
            .enumerate()
            .map(|(index, embedding)| (index, self.cosine_similarity(source_embedding, embedding)))
            .max_by(|&(_, a), &(_, b)| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(index, similarity)| {
                if similarity >= self.similarity_threshold {
                    Some(index)
                } else {
                    None
                }
            })
            .flatten()
    }

    fn flatten_object(&self, obj: &serde_json::Value) -> HashMap<String, serde_json::Value> {
        let mut flat_map = HashMap::new();
        self.flatten_recursive(obj, String::new(), &mut flat_map);
        flat_map
    }

    fn flatten_recursive(&self, obj: &serde_json::Value, prefix: String, flat_map: &mut HashMap<String, serde_json::Value>) {
        match obj {
            serde_json::Value::Object(map) => {
                for (key, value) in map {
                    let new_prefix = if prefix.is_empty() {
                        key.to_string()
                    } else {
                        format!("{}.{}", prefix, key)
                    };
                    self.flatten_recursive(value, new_prefix, flat_map);
                }
            }
            _ => {
                flat_map.insert(prefix, obj.clone());
            }
        }
    }

    fn unflatten_object(&self, flat_obj: &HashMap<String, serde_json::Value>) -> serde_json::Value {
        let mut result = serde_json::Map::new();
        for (key, value) in flat_obj {
            let mut current = &mut result;
            let parts: Vec<&str> = key.split('.').collect();
            for (i, part) in parts.iter().enumerate() {
                if i == parts.len() - 1 {
                    current.insert(part.to_string(), value.clone());
                } else {
                    current = current
                        .entry((*part).to_string())
                        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()))
                        .as_object_mut()
                        .unwrap();
                }
            }
        }
        serde_json::Value::Object(result)
    }

    pub async fn shapeshift(&self, source_obj: serde_json::Value, target_obj: serde_json::Value) -> serde_json::Value {
        println!("Starting shapeshift method");

        // Flatten both source and target objects
        let flat_source = self.flatten_object(&source_obj);
        let flat_target = self.flatten_object(&target_obj);

        println!("Flattened source: {:?}", flat_source);
        println!("Flattened target: {:?}", flat_target);

        // Extract keys from flattened objects
        let source_keys: Vec<String> = flat_source.keys().cloned().collect();
        let target_keys: Vec<String> = flat_target.keys().cloned().collect();

        println!("Source keys: {:?}", source_keys);
        println!("Target keys: {:?}", target_keys);

        // Calculate embeddings for the flattened keys
        let source_embeddings = self.calculate_embeddings(source_keys.clone()).await;
        let target_embeddings = self.calculate_embeddings(target_keys.clone()).await;

        println!("Source embeddings length: {}", source_embeddings.len());
        println!("Target embeddings length: {}", target_embeddings.len());

        // Create a new serde_json::Value to store the transformed key-value pairs
        let mut transformed = serde_json::Value::Object(serde_json::Map::new());

        // Helper function to insert nested keys
        fn insert_nested(value: &mut serde_json::Value, key: &str, new_value: serde_json::Value) {
            let parts: Vec<&str> = key.split('.').collect();
            let mut current = value;
            for (i, part) in parts.iter().enumerate() {
                if i == parts.len() - 1 {
                    if let Some(obj) = current.as_object_mut() {
                        obj.insert(part.to_string(), new_value.clone());
                    } else {
                        *current = serde_json::json!({ part.to_string(): new_value });
                    }
                } else {
                    if !current.is_object() {
                        *current = serde_json::json!({});
                    }
                    current = current.as_object_mut().unwrap().entry(part.to_string()).or_insert(serde_json::json!({}));
                }
            }
        }

        // Create a HashSet to keep track of used source keys
        let mut used_source_keys = HashSet::new();

        // Find closest matches and transform the source object
        for (target_idx, target_key) in target_keys.iter().enumerate() {
            println!("Processing target key: {} (index: {})", target_key, target_idx);
            if target_idx < target_embeddings.len() {
                let closest_match_idx = self.find_closest_match(&target_embeddings[target_idx], &source_embeddings);

                if let Some(idx) = closest_match_idx {
                    let source_key = &source_keys[idx];
                    if !used_source_keys.contains(source_key) {
                        println!("Matched source key: {} to target key: {}", source_key, target_key);
                        let value = if source_key.contains('.') {
                            // Handle nested source keys
                            let parts: Vec<&str> = source_key.split('.').collect();
                            let mut current = &flat_source[parts[0]];
                            for part in &parts[1..] {
                                current = &current[part];
                            }
                            current.clone()
                        } else {
                            flat_source.get(source_key).cloned().unwrap_or(serde_json::Value::Null)
                        };
                        insert_nested(&mut transformed, target_key, value);
                        used_source_keys.insert(source_key.clone());
                    } else {
                        println!("Source key already used: {}", source_key);
                        insert_nested(&mut transformed, target_key, serde_json::Value::Null);
                    }
                } else {
                    println!("No match found for target key: {}", target_key);
                    insert_nested(&mut transformed, target_key, serde_json::Value::Null);
                }
            } else {
                println!("Target index out of bounds for key: {}", target_key);
                insert_nested(&mut transformed, target_key, serde_json::Value::Null);
            }
        }

        println!("Transformed value: {:?}", transformed);

        // Create a JSON object with debug information and the result
        json!({
            "result": transformed,
            "debug_info": {
                "source_keys": source_keys,
                "target_keys": target_keys,
                "source_embeddings": source_embeddings,
                "target_embeddings": target_embeddings
            }
        })
    }
}

fn get_embeddings(texts: &[String]) -> Vec<Vec<f64>> {
    // Mock implementation: return embeddings based on input texts
    texts.iter().map(|text| {
        match text.as_str() {
            "name" | "full_name" => vec![0.9, 0.1, 0.0, 0.0, 0.0],
            "age" | "years_old" => vec![0.0, 0.9, 0.1, 0.0, 0.0],
            "city" | "location.city" => vec![0.0, 0.0, 0.9, 0.1, 0.0],
            "country" | "location.country" => vec![0.0, 0.0, 0.1, 0.9, 0.0],
            "location" => vec![0.0, 0.0, 0.7, 0.3, 0.0],
            _ => vec![0.2, 0.2, 0.2, 0.2, 0.2],
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_flatten_and_unflatten() {
        let shapeshift = Shapeshift::new("".to_string(), "".to_string(), "".to_string(), 0.8);
        let source_obj = json!({
            "name": "John Doe",
            "age": 30,
            "address": {
                "street": "123 Main St",
                "city": "Anytown",
                "country": "USA"
            }
        });

        let flattened = shapeshift.flatten_object(&source_obj);
        let unflattened = shapeshift.unflatten_object(&flattened);

        assert_eq!(source_obj, unflattened);
    }

    #[test]
    fn test_cosine_similarity() {
        let shapeshift = Shapeshift::new("".to_string(), "".to_string(), "".to_string(), 0.8);
        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![0.0, 1.0, 0.0];
        let similarity = shapeshift.cosine_similarity(&vec1, &vec2);
        assert_eq!(similarity, 0.0);
    }

    #[test]
    fn test_find_closest_match() {
        let shapeshift = Shapeshift::new("".to_string(), "".to_string(), "".to_string(), 0.8);
        let source_embedding = vec![1.0, 0.0, 0.0];
        let target_embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let closest_match = shapeshift.find_closest_match(&source_embedding, &target_embeddings);
        assert_eq!(closest_match, Some(0));
    }

    #[tokio::test]
    async fn test_shapeshift() {
        let shapeshift = Shapeshift::new("".to_string(), "".to_string(), "".to_string(), 0.95);
        let source_obj = json!({
            "name": "John Doe",
            "age": 30,
            "city": "New York"
        });
        let target_obj = json!({
            "full_name": "",
            "years_old": 0,
            "location": {
                "city": "",
                "country": ""
            }
        });

        let result = shapeshift.shapeshift(source_obj.clone(), target_obj.clone()).await;

        println!("Source object: {:?}", source_obj);
        println!("Target object: {:?}", target_obj);
        println!("Result: {:?}", result);

        let expected = json!({
            "full_name": "John Doe",
            "years_old": 30,
            "location": {
                "city": "New York",
                "country": null
            }
        });

        assert_eq!(result["result"], expected, "Shapeshift result does not match expected output");

        // Detailed assertions
        assert_eq!(result["result"]["full_name"], "John Doe", "full_name mismatch");
        assert_eq!(result["result"]["years_old"], 30, "years_old mismatch");
        assert_eq!(result["result"]["location"]["city"], "New York", "location.city mismatch");
        assert!(result["result"]["location"]["country"].is_null(), "location.country should be null");

        // Check debug info
        assert!(result["debug_info"]["source_keys"].is_array(), "source_keys should be an array");
        assert!(result["debug_info"]["target_keys"].is_array(), "target_keys should be an array");
        assert!(result["debug_info"]["source_embeddings"].is_array(), "source_embeddings should be an array");
        assert!(result["debug_info"]["target_embeddings"].is_array(), "target_embeddings should be an array");
    }
}