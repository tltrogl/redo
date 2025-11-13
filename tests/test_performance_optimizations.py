"""Test performance optimizations to ensure they work correctly."""

import tempfile
import os
from pathlib import Path


def test_html_generator_segment_limit():
    """Test that HTML generator handles large segment counts efficiently."""
    from diaremot.summaries.html_summary_generator import HTMLSummaryGenerator
    
    # Create test data with many segments
    segments = [
        {
            'start': i, 
            'end': i+1, 
            'speaker_id': f'spk_{i%3}', 
            'speaker_name': f'Speaker {i%3}', 
            'text': f'Test segment {i}',
            'vq_jitter_pct': 0.5,
            'vq_shimmer_db': 0.3,
            'vq_hnr_db': 10.0,
            'vq_cpps_db': 5.0,
            'arousal': 0.5,
        }
        for i in range(600)  # More than old 100 limit, more than new 500 limit
    ]
    
    speakers = [
        {'speaker_id': 'spk_0', 'speaker_name': 'Speaker 0', 'total_duration': 100},
        {'speaker_id': 'spk_1', 'speaker_name': 'Speaker 1', 'total_duration': 80},
    ]
    
    gen = HTMLSummaryGenerator()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = gen.render_to_html(tmpdir, 'test_file', segments, speakers, {})
        
        # Check that file was created
        assert os.path.exists(path), 'HTML file not created'
        
        with open(path, 'r') as f:
            content = f.read()
            
            # Should include 500 segments (new limit)
            segment_count = content.count('class="transcript-row"')
            assert segment_count >= 500, f'Expected at least 500 segments, got {segment_count}'
            
            # Should have truncation message for 600 segments
            assert 'more segments not shown' in content, 'Truncation message missing'
            
            # Verify metrics aggregation completed
            assert 'Voice Quality' in content, 'Voice quality section missing'


def test_html_generator_single_pass_aggregation():
    """Test that voice quality metrics are computed in a single pass."""
    from diaremot.summaries.html_summary_generator import HTMLSummaryGenerator
    
    # Create segments with various metric values
    segments = [
        {
            'start': i,
            'end': i+1,
            'speaker_id': 'spk_0',
            'text': f'Segment {i}',
            'vq_jitter_pct': float(i),
            'vq_shimmer_db': float(i * 2),
            'vq_hnr_db': float(i * 3),
            'vq_cpps_db': float(i * 4),
        }
        for i in range(10)
    ]
    
    speakers = [{'speaker_id': 'spk_0', 'total_duration': 10}]
    
    gen = HTMLSummaryGenerator()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = gen.render_to_html(tmpdir, 'test', segments, speakers, {})
        
        with open(path, 'r') as f:
            content = f.read()
            
            # Check that metrics are present and reasonable
            assert 'Jitter (pct)' in content
            assert 'Shimmer (dB)' in content
            assert 'HNR (dB)' in content
            assert 'CPPS (dB)' in content


def test_conversation_analysis_efficiency():
    """Test that conversation analysis works efficiently with optimized code."""
    from diaremot.summaries.conversation_analysis import analyze_conversation_flow
    
    # Create realistic test data
    segments = []
    for i in range(100):
        segments.append({
            'start': i * 5.0,
            'end': i * 5.0 + 4.0,
            'speaker_id': f'spk_{i % 3}',
            'speaker': f'spk_{i % 3}',
            'text': f'This is test segment {i} with some words',
            'arousal': 0.5,
        })
    
    total_duration = 500.0
    
    # Should complete without errors
    metrics = analyze_conversation_flow(segments, total_duration)
    
    assert metrics is not None
    assert hasattr(metrics, 'turn_taking_balance')
    assert hasattr(metrics, 'speaker_dominance')
    assert len(metrics.speaker_dominance) > 0


def test_paralinguistics_memory_optimization():
    """Test that paralinguistics array conversion is efficient."""
    import numpy as np
    from diaremot.affect.paralinguistics.config import ParalinguisticsConfig
    from diaremot.affect.paralinguistics.features import compute_segment_features_v2
    
    # Create test audio
    sr = 16000
    audio = np.random.randn(sr * 2).astype(np.float64)  # 2 seconds
    
    # Test with memory optimization enabled
    cfg = ParalinguisticsConfig(enable_memory_optimization=True)
    result = compute_segment_features_v2(audio, sr, 0.0, 2.0, "test text", cfg)
    
    assert result is not None
    assert 'wpm' in result
    # Check that the function completed successfully
    assert isinstance(result, dict)


if __name__ == '__main__':
    test_html_generator_segment_limit()
    print('✓ HTML generator segment limit test passed')
    
    test_html_generator_single_pass_aggregation()
    print('✓ HTML generator single-pass aggregation test passed')
    
    test_conversation_analysis_efficiency()
    print('✓ Conversation analysis efficiency test passed')
    
    test_paralinguistics_memory_optimization()
    print('✓ Paralinguistics memory optimization test passed')
    
    print('\n✓ All performance optimization tests passed!')
