# Lab 5 - CI/CD with GitHub Actions

## Answers

1. A unit test should not load the real MLflow model because the server may be
   offline or unreachable, credentials and registry state may differ, the
   `champion` alias may move, and downloading/loading the model is slow and can
   fail for reasons unrelated to the code under test.

2. The fake-model API test proves that the endpoint accepts and decodes an
   uploaded image, preprocesses it, calls the model interface, maps the winning
   output to the correct class, and returns valid JSON. It does not prove that a
   trained model is accurate, that MLflow is reachable, or that the production
   model artifact can be loaded.

3. The cross-split content-hash check caught the copied image. It also exposed
   43 pre-existing cross-split duplicates, which were removed before the
   deliberate test. A duplicate is silent data leakage: evaluation results
   become optimistically biased while the pipeline appears healthy. A corrupt
   image fails loudly and is easier to detect before trusting any metric.

4. The contract can run before data is pushed to DVC (locally or in the data
   preparation pipeline), and in CI after `dvc pull` when the runner has access
   to a remote through repository secrets. It can also run in a self-hosted
   runner that already has authorized DVC access.

5. `needs: test` skips `smoke` when `test` fails. Without it, both jobs start
   independently and the image may be built even though unit tests are red.

6. The measured GitHub run durations are recorded after the workflow runs.
   Later runs should benefit from uv and Docker/GitHub Actions caches, mainly in
   dependency synchronization and image build layers.

7. A commit SHA is immutable and traceable to exact source. With `latest`, one
   machine can pull before a new release and another afterward, leaving both
   machines labeled `latest` while running different code.

8. Explicit permissions apply least privilege: the publishing job needs package
   write access but only repository read access. An unnecessarily broad token in
   a compromised step could modify code, workflows, releases, issues, secrets,
   or other repository resources allowed by that token.

9. The demonstration pull request and branch-protection result are recorded
   after configuration. Required checks give the team an enforceable, shared,
   reproducible gate instead of relying on each developer's laptop and memory.

10. Version 2 (learning rate `0.000001`, one epoch) scored 0.0896 and was
    assigned `challenger`. Version 3 (learning rate `0.001`, five epochs)
    scored 0.2614 and was also assigned `challenger`, replacing that alias.
    Version 1 remained `champion` at 0.3120. With 1,055 evaluation images,
    `2 * se` was 0.0285, so promotion required an accuracy of at least 0.3406.

11. Promotion can run on a self-hosted GitHub Actions runner on the MLflow
    machine, or against a reachable hosted registry such as DagsHub MLflow.
    Another option is a local scheduler/webhook triggered when training ends.

12. Pipeline:

```text
git push
   |
   +--> [automatic] lint + unit tests
           |
           +--> [automatic] build image + SKIP_MODEL health smoke test
                   |
                   +--> main only: [automatic] publish image to GHCR
                                      code version = git commit SHA

training run --> [automatic metrics/model logging] MLflow registry candidate
   |
   +--> [human or training automation] run promote.py quality gate
           |
           +--> model version = MLflow champion alias

[human/deployment platform] deploy the SHA-tagged image and restart inference
to load the current champion model.
```
