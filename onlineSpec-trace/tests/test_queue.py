from onlinespec_trace.queue import build_jobs
from onlinespec_trace.core.io_utils import write_jsonl
import pytest

def test_queue_short_datasets_first_and_no_duplicate_pairs(tmp_path):
    short=tmp_path/"short.jsonl";long=tmp_path/"long.jsonl"
    write_jsonl(short,[dict(question_id=0,turns=["a"])])
    write_jsonl(long,[dict(question_id=i,turns=["a"]) for i in range(4)])
    spec=dict(output_root=str(tmp_path/"results"),models=[dict(name="m",profile="deepseek",target="t",draft="d")],
        datasets=[dict(name="long",path=str(long)),dict(name="short",path=str(short),expected_questions=1)],seeds=[0,1])
    jobs=build_jobs(spec)
    assert [j["questions"] for j in jobs]==[1,1,4,4]
    assert len({j["settings"]["output"] for j in jobs})==4
    spec["seeds"]=[0,0]
    with pytest.raises(ValueError,match="Duplicate"):
        build_jobs(spec)
