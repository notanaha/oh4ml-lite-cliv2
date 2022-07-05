# oh4ml-lite-cliv2
Demonstrates cliv2 capability on Azure ML
<i><ul>
  <li>az ml environment create --file 01_createEnvironment.yml</li>
  <li>az ml job create -n 02_trainOnRemtoe_01 -f 02_trainOnRemote.yml</li>
  <li>az ml job download --name 02_trainOnRemtoe_01 --download-path ./ --all</li>
  <li>az ml model create --file 03_createModel.yml</li>
  <li>az ml online-endpoint create --name ep-arima-inference-01 -f endpoints/ep-arima-inference-01.yml</li>
  <li>az ml online-deployment create --name blue --endpoint ep-arima-inference-01 -f endpoints/blue-arima-inference-01.yml --all-traffic</li>
  <li>az ml online-endpoint invoke --name ep-arima-inference-01 --request-file endpoints/sample-request.json</li>
</ul></i>
<br>
