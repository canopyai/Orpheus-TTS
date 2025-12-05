"""
Audio Quality Validation Framework for FP8 Orpheus-TTS
Comprehensive validation tools for assessing FP8 model performance and audio quality
"""

import os
import numpy as np
import torch
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import whisper
import librosa
import librosa.display
from scipy.signal import correlate
from transformers import AutoTokenizer
import json
import pandas as pd
from datetime import datetime


class AudioQualityValidator:
    """
    Comprehensive validation framework for comparing FP8 and BF16 model outputs.
    """
    
    def __init__(self, whisper_model="base", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the validator with necessary models.
        
        Args:
            whisper_model: Size of Whisper model for ASR ("base", "small", "medium", "large")
            device: Device to run models on
        """
        self.device = device
        self.whisper_model = whisper.load_model(whisper_model, device=device)
        self.results_dir = Path("validation_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def calculate_wer(self, reference_text: str, hypothesis_text: str) -> float:
        """
        Calculate Word Error Rate between reference and hypothesis text.
        
        Args:
            reference_text: Original text
            hypothesis_text: ASR transcribed text
            
        Returns:
            WER score (0.0 = perfect, 1.0 = completely wrong)
        """
        # Tokenize and normalize
        ref_words = reference_text.lower().split()
        hyp_words = hypothesis_text.lower().split()
        
        # Dynamic programming for edit distance
        d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
        
        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j
            
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(
                        d[i-1][j] + 1,    # Deletion
                        d[i][j-1] + 1,    # Insertion
                        d[i-1][j-1] + 1   # Substitution
                    )
        
        wer = d[len(ref_words)][len(hyp_words)] / len(ref_words) if ref_words else 0.0
        return wer
    
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio using Whisper ASR.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        result = self.whisper_model.transcribe(audio_path, language="en")
        return result["text"].strip()
    
    def analyze_audio_quality(self, audio_path: str) -> Dict:
        """
        Analyze various audio quality metrics.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary of audio quality metrics
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Calculate metrics
        metrics = {}
        
        # Signal-to-Noise Ratio (estimated)
        # Using a simple method: ratio of signal power to silence power
        silence_threshold = np.percentile(np.abs(audio), 10)
        signal_power = np.mean(audio[np.abs(audio) > silence_threshold]**2)
        noise_power = np.mean(audio[np.abs(audio) <= silence_threshold]**2)
        metrics['snr_db'] = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        # Spectral centroid (brightness indicator)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        metrics['mean_spectral_centroid'] = np.mean(spectral_centroids)
        metrics['std_spectral_centroid'] = np.std(spectral_centroids)
        
        # Zero crossing rate (noise/artifacts indicator)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        metrics['mean_zcr'] = np.mean(zcr)
        metrics['std_zcr'] = np.std(zcr)
        
        # RMS energy
        rms = librosa.feature.rms(y=audio)[0]
        metrics['mean_rms'] = np.mean(rms)
        metrics['std_rms'] = np.std(rms)
        
        # Pitch statistics (using librosa's pitch tracking)
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        # Get pitch values where magnitude is significant
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if pitch_values:
            metrics['mean_pitch'] = np.mean(pitch_values)
            metrics['std_pitch'] = np.std(pitch_values)
        else:
            metrics['mean_pitch'] = 0
            metrics['std_pitch'] = 0
            
        return metrics
    
    def compare_spectrograms(self, audio_path1: str, audio_path2: str, 
                           save_path: Optional[str] = None) -> Dict:
        """
        Compare spectrograms of two audio files.
        
        Args:
            audio_path1: Path to first audio file (e.g., FP8)
            audio_path2: Path to second audio file (e.g., BF16)
            save_path: Optional path to save comparison plot
            
        Returns:
            Dictionary with comparison metrics
        """
        # Load audio files
        audio1, sr1 = librosa.load(audio_path1, sr=None)
        audio2, sr2 = librosa.load(audio_path2, sr=None)
        
        # Ensure same sample rate
        if sr1 != sr2:
            audio2 = librosa.resample(audio2, orig_sr=sr2, target_sr=sr1)
            sr2 = sr1
        
        # Compute spectrograms
        D1 = librosa.stft(audio1)
        D2 = librosa.stft(audio2)
        
        DB1 = librosa.amplitude_to_db(np.abs(D1), ref=np.max)
        DB2 = librosa.amplitude_to_db(np.abs(D2), ref=np.max)
        
        # Ensure same shape for comparison
        min_frames = min(DB1.shape[1], DB2.shape[1])
        DB1 = DB1[:, :min_frames]
        DB2 = DB2[:, :min_frames]
        
        # Calculate difference
        diff = DB1 - DB2
        
        # Metrics
        metrics = {
            'mean_absolute_difference': np.mean(np.abs(diff)),
            'max_absolute_difference': np.max(np.abs(diff)),
            'rmse': np.sqrt(np.mean(diff**2)),
            'correlation': np.corrcoef(DB1.flatten(), DB2.flatten())[0, 1]
        }
        
        # Plot if requested
        if save_path:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Original spectrograms
            librosa.display.specshow(DB1, sr=sr1, x_axis='time', y_axis='hz', ax=axes[0, 0])
            axes[0, 0].set_title('FP8 Model Spectrogram')
            axes[0, 0].set_colorbar(format='%+2.0f dB')
            
            librosa.display.specshow(DB2, sr=sr1, x_axis='time', y_axis='hz', ax=axes[0, 1])
            axes[0, 1].set_title('BF16 Model Spectrogram')
            axes[0, 1].set_colorbar(format='%+2.0f dB')
            
            # Difference
            im = librosa.display.specshow(diff, sr=sr1, x_axis='time', y_axis='hz', ax=axes[1, 0])
            axes[1, 0].set_title('Difference (FP8 - BF16)')
            fig.colorbar(im, ax=axes[1, 0], format='%+2.0f dB')
            
            # Histogram of differences
            axes[1, 1].hist(diff.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
            axes[1, 1].set_xlabel('Difference (dB)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Distribution of Spectral Differences')
            axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        return metrics
    
    def analyze_attention_alignment(self, model, text: str, voice: str, 
                                  save_path: Optional[str] = None) -> Dict:
        """
        Analyze attention patterns for text-to-speech alignment.
        Note: This requires access to model internals, which may need modification.
        """
        # This is a placeholder for attention analysis
        # In practice, you'd need to hook into the model's attention layers
        print("Note: Attention analysis requires model modification to expose attention weights.")
        return {"status": "not_implemented"}
    
    def validate_emotional_tags(self, audio_paths: Dict[str, str], 
                              emotions: List[str]) -> Dict:
        """
        Validate that emotional tags are properly rendered in audio.
        
        Args:
            audio_paths: Dictionary mapping emotion to audio file path
            emotions: List of emotions to validate
            
        Returns:
            Validation results
        """
        results = {}
        
        for emotion in emotions:
            if emotion not in audio_paths:
                continue
                
            audio_path = audio_paths[emotion]
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Extract features that correlate with emotions
            features = {}
            
            # Pitch variation (excitement, sadness)
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
                features['pitch_variance'] = np.var(pitch_values)
            
            # Energy variation (laughter, sighs)
            rms = librosa.feature.rms(y=audio)[0]
            features['energy_variance'] = np.var(rms)
            features['energy_range'] = np.max(rms) - np.min(rms)
            
            # Tempo/rhythm (for laugh detection)
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            features['tempo'] = tempo
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            features['spectral_brightness'] = np.mean(spectral_centroids)
            
            results[emotion] = features
            
        return results
    
    def generate_validation_report(self, fp8_results: Dict, bf16_results: Dict, 
                                 output_path: str = "validation_report.json") -> None:
        """
        Generate a comprehensive validation report.
        
        Args:
            fp8_results: Results from FP8 model validation
            bf16_results: Results from BF16 model validation
            output_path: Path to save the report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "fp8_results": fp8_results,
            "bf16_results": bf16_results,
            "comparison": {},
            "recommendations": []
        }
        
        # Calculate comparisons
        if "wer" in fp8_results and "wer" in bf16_results:
            wer_increase = (fp8_results["wer"] - bf16_results["wer"]) / bf16_results["wer"] * 100
            report["comparison"]["wer_increase_percent"] = wer_increase
            
            if wer_increase > 5:
                report["recommendations"].append(
                    "WER increased by more than 5%. Consider adjusting calibration dataset or quantization parameters."
                )
        
        # Audio quality comparison
        if "audio_metrics" in fp8_results and "audio_metrics" in bf16_results:
            fp8_metrics = fp8_results["audio_metrics"]
            bf16_metrics = bf16_results["audio_metrics"]
            
            snr_diff = fp8_metrics.get("snr_db", 0) - bf16_metrics.get("snr_db", 0)
            report["comparison"]["snr_difference_db"] = snr_diff
            
            if snr_diff < -3:
                report["recommendations"].append(
                    "SNR decreased by more than 3dB. Check for quantization-induced noise."
                )
        
        # Save report
        report_path = self.results_dir / output_path
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"Validation report saved to: {report_path}")
        
        # Also create a human-readable summary
        summary_path = self.results_dir / "validation_summary.txt"
        with open(summary_path, "w") as f:
            f.write("Orpheus-TTS FP8 Validation Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {report['timestamp']}\n\n")
            
            if "comparison" in report:
                f.write("Key Metrics:\n")
                for key, value in report["comparison"].items():
                    f.write(f"  - {key}: {value:.2f}\n")
            
            if report["recommendations"]:
                f.write("\nRecommendations:\n")
                for rec in report["recommendations"]:
                    f.write(f"  - {rec}\n")
        
        print(f"Summary saved to: {summary_path}")


class TestSuite:
    """
    Comprehensive test suite for FP8 validation.
    """
    
    def __init__(self, validator: AudioQualityValidator):
        self.validator = validator
        self.test_sentences = self._load_test_sentences()
        
    def _load_test_sentences(self) -> List[Dict]:
        """Load comprehensive test sentences."""
        return [
            # Basic clarity tests
            {"text": "The quick brown fox jumps over the lazy dog.", "type": "pangram"},
            {"text": "Pack my box with five dozen liquor jugs.", "type": "pangram"},
            
            # Phonetic challenges
            {"text": "She sells seashells by the seashore.", "type": "tongue_twister"},
            {"text": "Peter Piper picked a peck of pickled peppers.", "type": "tongue_twister"},
            
            # Numbers and symbols
            {"text": "The meeting is at 3:45 PM on December 21st, 2024.", "type": "datetime"},
            {"text": "Call me at 555-1234 or email john@example.com.", "type": "contact"},
            {"text": "The temperature is 72°F with 65% humidity.", "type": "measurements"},
            
            # Emotional content
            {"text": "<laugh> That's the funniest thing I've heard all day!", "type": "emotion_laugh"},
            {"text": "<sigh> I'm feeling a bit tired today.", "type": "emotion_sigh"},
            {"text": "<chuckle> Well, that's interesting!", "type": "emotion_chuckle"},
            
            # Long sentences
            {"text": "In the midst of winter, I found there was, within me, an invincible summer, and that makes me happy.", "type": "long_philosophical"},
            {"text": "The conference will cover artificial intelligence, machine learning, quantum computing, and biotechnology.", "type": "long_technical"},
            
            # Punctuation variations
            {"text": "Wait! Can you repeat that? I didn't quite catch it.", "type": "punctuation"},
            {"text": "Hmm... Let me think about that for a moment.", "type": "punctuation"},
        ]
    
    def run_comprehensive_test(self, fp8_model, bf16_model, voices: List[str] = ["tara"]) -> Dict:
        """
        Run comprehensive validation tests.
        
        Args:
            fp8_model: FP8 quantized model
            bf16_model: Original BF16 model
            voices: List of voices to test
            
        Returns:
            Comprehensive test results
        """
        results = {
            "fp8": {"per_sentence": [], "aggregate": {}},
            "bf16": {"per_sentence": [], "aggregate": {}},
            "comparison": {}
        }
        
        for voice in voices:
            print(f"\nTesting voice: {voice}")
            
            for i, test in enumerate(self.test_sentences):
                print(f"  Test {i+1}/{len(self.test_sentences)}: {test['type']}")
                
                # Generate audio with both models
                fp8_audio_path = f"temp_fp8_{voice}_{i}.wav"
                bf16_audio_path = f"temp_bf16_{voice}_{i}.wav"
                
                # FP8 generation
                self._generate_and_save(fp8_model, test['text'], voice, fp8_audio_path)
                
                # BF16 generation
                self._generate_and_save(bf16_model, test['text'], voice, bf16_audio_path)
                
                # Analyze both outputs
                fp8_result = self._analyze_single_output(test, fp8_audio_path, voice)
                bf16_result = self._analyze_single_output(test, bf16_audio_path, voice)
                
                # Compare spectrograms
                comparison = self.validator.compare_spectrograms(
                    fp8_audio_path, 
                    bf16_audio_path,
                    save_path=f"spectrogram_comparison_{voice}_{i}.png"
                )
                
                # Store results
                results["fp8"]["per_sentence"].append(fp8_result)
                results["bf16"]["per_sentence"].append(bf16_result)
                
                # Clean up temp files
                os.remove(fp8_audio_path)
                os.remove(bf16_audio_path)
        
        # Calculate aggregate metrics
        results["fp8"]["aggregate"] = self._calculate_aggregates(results["fp8"]["per_sentence"])
        results["bf16"]["aggregate"] = self._calculate_aggregates(results["bf16"]["per_sentence"])
        
        # Generate final report
        self.validator.generate_validation_report(
            results["fp8"]["aggregate"],
            results["bf16"]["aggregate"]
        )
        
        return results
    
    def _generate_and_save(self, model, text: str, voice: str, output_path: str):
        """Generate audio and save to file."""
        audio_chunks = []
        for chunk in model.generate_speech(prompt=text, voice=voice):
            audio_chunks.append(chunk)
        
        audio_bytes = b''.join(audio_chunks)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32767.0
        
        sf.write(output_path, audio_float, 24000, subtype='PCM_16')
    
    def _analyze_single_output(self, test: Dict, audio_path: str, voice: str) -> Dict:
        """Analyze a single audio output."""
        result = {
            "test_type": test["type"],
            "original_text": test["text"],
            "voice": voice
        }
        
        # Transcribe
        transcribed = self.validator.transcribe_audio(audio_path)
        result["transcribed_text"] = transcribed
        
        # Calculate WER
        # Remove emotion tags for WER calculation
        clean_text = test["text"]
        for tag in ["<laugh>", "<sigh>", "<chuckle>", "<cough>", "<pause>"]:
            clean_text = clean_text.replace(tag, "").strip()
        
        result["wer"] = self.validator.calculate_wer(clean_text, transcribed)
        
        # Audio quality metrics
        result["audio_metrics"] = self.validator.analyze_audio_quality(audio_path)
        
        return result
    
    def _calculate_aggregates(self, results: List[Dict]) -> Dict:
        """Calculate aggregate metrics from individual results."""
        aggregates = {
            "wer": np.mean([r["wer"] for r in results]),
            "wer_by_type": {},
            "audio_metrics": {}
        }
        
        # WER by type
        types = set(r["test_type"] for r in results)
        for test_type in types:
            type_results = [r for r in results if r["test_type"] == test_type]
            aggregates["wer_by_type"][test_type] = np.mean([r["wer"] for r in type_results])
        
        # Average audio metrics
        metric_keys = results[0]["audio_metrics"].keys() if results else []
        for key in metric_keys:
            values = [r["audio_metrics"][key] for r in results if key in r["audio_metrics"]]
            aggregates["audio_metrics"][key] = np.mean(values)
        
        return aggregates


if __name__ == "__main__":
    # Example usage
    print("Orpheus-TTS FP8 Audio Quality Validator")
    print("=" * 50)
    
    # Initialize validator
    validator = AudioQualityValidator(whisper_model="base")
    
    # Example: Compare two audio files
    if os.path.exists("fp8_sample.wav") and os.path.exists("bf16_sample.wav"):
        print("\nComparing FP8 and BF16 audio samples...")
        
        # Transcribe both
        fp8_text = validator.transcribe_audio("fp8_sample.wav")
        bf16_text = validator.transcribe_audio("bf16_sample.wav")
        
        print(f"\nFP8 transcription: {fp8_text}")
        print(f"BF16 transcription: {bf16_text}")
        
        # Compare spectrograms
        comparison = validator.compare_spectrograms(
            "fp8_sample.wav",
            "bf16_sample.wav", 
            save_path="spectrogram_comparison.png"
        )
        
        print("\nSpectrogram comparison:")
        for key, value in comparison.items():
            print(f"  {key}: {value:.4f}")
    else:
        print("\nNo sample audio files found. Run inference examples first.")
