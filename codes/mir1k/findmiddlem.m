clear all
winsize = 1024; nFFT = 1024; hop = 512; scf = 1;
windows = sin(0:pi/winsize:pi-pi/winsize);

c1 = 0.4:0.01:1;c2 = 0.4:0.01:1;
k =1;
for i1 = 1:numel(c1)
    i1
    for i2 = 1:numel(c1)
        i2
        if c1(i1)<=c2(i2)
            file = dir(['hesheng\hao',filesep,'*wav']);
for i = 1:numel(file)
  wavfile = wavread(['hesheng\hao',filesep,file(i).name]);
  wavfile=wavfile./sqrt(sum(wavfile.^2));
  spectrum = scf * stft(wavfile, nFFT ,windows, hop, 16000);
  spectrum=abs(spectrum);
  t1 = find(spectrum>=c1(i1)); 
  t2 = find(spectrum(t1)<=c2(i2));
  counter(i)=numel(t2)/numel(spectrum);
end
     mi=min(counter);
file = dir(['hesheng\bu',filesep,'*wav']);
for i = 1:numel(file)
  wavfile = wavread(['hesheng\bu',filesep,file(i).name]);
  wavfile=wavfile./sqrt(sum(wavfile.^2));
  spectrum = scf * stft(wavfile, nFFT ,windows, hop, 16000);
  spectrum=abs(spectrum);
  t1 = find(spectrum>=c1(i1)); 
  t2 = find(spectrum(t1)<=c2(i2));
  counter(i)=numel(t2)/numel(spectrum);
end
    ma =max(counter);
    if mi>ma
        k
        best(k,1) = c1(i1);
        best(k,2) = c2(i2);
        k = k+1;
    end
        end
    end
end