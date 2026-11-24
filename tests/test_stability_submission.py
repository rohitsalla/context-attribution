import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import submit_stability


class SubmissionTests(unittest.TestCase):
    def test_both_smokes_gate_serial_chain_and_rerun_is_idempotent(self):
        original=Path.cwd();jobs=[];releases=[]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'data/stability').mkdir(parents=True);(root/'data/stability/manifest.json').write_text('{}')
            (root/'research/v5').mkdir(parents=True)
            requirements={'python_major_minor':'3.10','torch':'2.10.0+cu128','transformers':'5.17.0','cuda':'12.8'}
            (root/'research/v5/STABILITY_RUNTIME.json').write_text(json.dumps(requirements))
            models=root/'models';models.mkdir();(models/'config.json').write_text('{}')
            paths=root/'paths.json';paths.write_text(json.dumps({'qwen7':str(models),'mistral7':str(models)}))
            base=root/'base_python';base.touch()
            python=root/'python';python.symlink_to(base)
            argv=['submit','--project',str(root),'--paths',str(paths),'--python-bin',str(python),
                  '--recomp-path',str(models),'--source-commit','fixed-source']
            def command(args,**kw):
                if args[0]==str(python):return json.dumps({**requirements,'prefix':str(root)})
                if args[0]=='sbatch':
                    jobs.append(args);return str(1000+len(jobs))+'\n'
                if args[0]=='squeue':return ''.join(f'{1000+i}|PENDING|JobHeldUser\n' for i in range(1,len(jobs)+1))
                raise AssertionError(args)
            try:
                with patch.object(sys,'argv',argv),patch.object(submit_stability.subprocess,'check_output',side_effect=command),patch.object(submit_stability.subprocess,'run',side_effect=lambda args,**kw:releases.append(args)):
                    submit_stability.main()
                    self.assertEqual(len(jobs),35);self.assertEqual(len(releases),1)
                    receipt=json.loads((root/'logs/stability_submission.json').read_text())
                    self.assertEqual(receipt['identity']['python_bin'],str(python))
                    self.assertIn('--dependency=afterany:1001',jobs[1])
                    self.assertIn('--dependency=afterok:1001,afterany:1002',jobs[2])
                    self.assertIn('--dependency=afterok:1002,afterany:1018',jobs[18])
                    self.assertEqual(len(releases[0][2:]),35)
                    submit_stability.main();self.assertEqual(len(jobs),35);self.assertEqual(len(releases),1)
                    def wrong_runtime(args,**kw):
                        if args[0]==str(python):return json.dumps({**requirements,'transformers':'wrong-environment'})
                        raise AssertionError('Runtime mismatch must stop before any Slurm call')
                    with patch.object(submit_stability.subprocess,'check_output',side_effect=wrong_runtime):
                        with self.assertRaises(RuntimeError):submit_stability.main()
                    self.assertEqual(len(jobs),35)
            finally:os.chdir(original)


if __name__=='__main__':unittest.main()
