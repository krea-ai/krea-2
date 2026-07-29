# Safety

We implemented safety measures across the full model development lifecycle. We applied targeted fine-tuning techniques to reduce the model’s susceptibility to generating harmful content in response to both direct and adversarial prompts, and we conducted multiple rounds of internal and external safety evaluation before release.

For Krea’s hosted products incorporating Krea 2 models, we deploy input and output classifiers using a combination of proprietary and third-party detection tools to flag or block policy-violating prompts and generated images.

## Downstream Deployment and Reporting

Because this is also an open-weights release, Krea does not control downstream deployment of the model. Under the Krea 2 Community License, deployers are required to implement content filtering measures or equivalent review processes to prevent the generation or distribution of unlawful or policy-violating content appropriate to their use case. Deployers who fail to implement required safeguards are in breach of the license. See the license for details.

Krea maintains reporting channels for harmful, illegal, or policy-violating outputs at safety@krea.ai. Reports involving potential CSAM are escalated to NCMEC as required by law. Krea reserves the right to update model weights or revoke access in response to identified misuse patterns.

## Safety Evaluation

We conducted multiple rounds of internal and external safety evaluations prior to release. Testing focused on the model's resilience to adversarial elicitation of harmful outputs, including child sexual abuse material (CSAM), non-consensual intimate imagery (NCII), sexually explicit content, and other policy-violating content. The final pre-release evaluation included approximately 2,500 prompts and compared the model against Flux 2 Dev, Flux Klein 9B, and Z Image across these risk categories. Based on these evaluations, the release checkpoint demonstrated high resilience against violative inputs across the tested risk categories.
